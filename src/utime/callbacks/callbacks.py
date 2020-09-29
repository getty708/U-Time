"""
References:
- https://github.com/perslev/MultiPlanarUNet/blob/master/mpunet/callbacks/callbacks.py
"""
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from queue import Queue
from threading import Thread

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import psutil

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from utime.utils import highlighted
from utime.logging import ScreenLogger


class Validation(Callback):
    """
    Validation computation callback.
    
    Samples a number of validation batches from a deepsleep
    ValidationMultiSequence object
    and computes for all tasks:
        - Batch-wise validation loss
        - Batch-wise metrics as specified in model.metrics_tensors
        - Epoch-wise pr-class and average precision
        - Epoch-wise pr-class and average recall
        - Epoch-wise pr-class and average dice coefficients
    ... and adds all results to the log dict

    Note: The purpose of this callback over the default tf.keras evaluation
    mechanism is to calculate certain metrics over the entire epoch of data as
    opposed to averaged batch-wise computations.
    """
    def __init__(self, val_sequence, steps, logger=None, verbose=True):
        """
        Args:
            val_sequence: A deepsleep ValidationMultiSequence object
            steps:        Numer of batches to sample from val_sequences in each
                          validation epoch for each validation set
            logger:       An instance of a MultiPlanar Logger that prints to
                          screen and/or file
            verbose:      Print progress to screen - OBS does not use Logger
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.sequences = val_sequence.sequences
        self.steps = steps
        self.verbose = verbose
        self.n_classes = val_sequence.n_classes
        self.IDs = val_sequence.IDs
        self.print_round = 3
        self.log_round = 4

    def predict(self):
        def eval(queue, steps, TPs, relevant, selected, id_, lock):
            step = 0
            while step < steps:
                # Get prediction and true labels from prediction queue
                step += 1
                p, y = queue.get(block=True)

                # Argmax and CM elements
                p = p.argmax(-1).ravel()
                y = y.ravel()

                # Compute relevant CM elements
                # We select the number following the largest class integer when
                # y != pred, then bincount and remove the added dummy class
                tps = np.bincount(np.where(y == p, y, self.n_classes),
                                  minlength=self.n_classes+1)[:-1]
                rel = np.bincount(y, minlength=self.n_classes)
                sel = np.bincount(p, minlength=self.n_classes)

                # Update counts on shared lists
                lock.acquire()
                TPs[id_] += tps.astype(np.uint64)
                relevant[id_] += rel.astype(np.uint64)
                selected[id_] += sel.astype(np.uint64)
                lock.release()

        # Get tensors to run and their names
        metrics = self.model.metrics
        metrics_names = self.model.metrics_names
        self.model.reset_metrics()
        assert "loss" in metrics_names and metrics_names.index("loss") == 0
        # assert len(metrics_names)-1 == len(metrics)
        assert len(metrics_names) == len(metrics) # FIXME: Why -1?

        # Prepare arrays for CM summary stats
        TPs, relevant, selected, metrics_results = {}, {}, {}, {}
        count_threads = []
        for id_, sequence in zip(self.IDs, self.sequences):
            # Add count arrays to the result dictionaries
            TPs[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            relevant[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)
            selected[id_] = np.zeros(shape=(self.n_classes,), dtype=np.uint64)

            # Fetch validation samples from the generator
            pool = ThreadPoolExecutor(max_workers=7)
            result = pool.map(sequence.__getitem__, np.arange(self.steps))

            # Prepare queue and thread for computing counts
            count_queue = Queue(maxsize=self.steps)
            count_thread = Thread(target=eval, args=[count_queue, self.steps,
                                                     TPs, relevant, selected,
                                                     id_, Lock()])
            count_threads.append(count_thread)
            count_thread.start()

            # Predict and evaluate on all batches
            for i, (X, y) in enumerate(result):
                if self.verbose:
                    s = "   {}Validation step: {}/{}".format(f"[{id_}] "
                                                             if id_ else "",
                                                             i+1, self.steps)
                    print(s, end="\r", flush=True)
                pred = self.model.predict_on_batch(X)
                # Put values in the queue for counting
                # keras .predict_on__batch returns np.ndarray.
                # count_queue.put([pred.numpy(), y])
                count_queue.put([pred, y])
                # Run all metrics
                for metric in metrics:
                    metric(y, pred)

            # Compute mean metrics for the dataset
            metrics_results[id_] = {}
            for metric, name in zip(metrics, metrics_names[1:]):
                metrics_results[id_][name] = metric.result().numpy()
            self.model.reset_metrics()
            pool.shutdown()
            self.logger("")
        self.logger("")
        # Terminate count threads
        for thread in count_threads:
            thread.join()
        return TPs, relevant, selected, metrics_results

    @staticmethod
    def _compute_dice(tp, rel, sel):
        # Get data masks (to avoid div. by zero warnings)
        # We set precision, recall, dice to 0 in for those particular cls.
        sel_mask = sel > 0
        rel_mask = rel > 0

        # prepare arrays
        precisions = np.zeros(shape=tp.shape, dtype=np.float32)
        recalls = np.zeros_like(precisions)
        dices = np.zeros_like(precisions)

        # Compute precisions, recalls
        precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
        recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]

        # Compute dice
        intrs = (2 * precisions * recalls)
        union = (precisions + recalls)
        dice_mask = union > 0
        dices[dice_mask] = intrs[dice_mask] / union[dice_mask]

        return precisions, recalls, dices

    def _print_val_results(self, precisions, recalls, dices, metrics, epoch,
                           name, classes):
        # Log the results
        # We add them to a pd dataframe just for the pretty print output
        index = ["cls %i" % i for i in classes]
        metric_keys, metric_vals = map(list, list(zip(*metrics.items())))
        col_order = metric_keys + ["precision", "recall", "dice"]
        nan_arr = np.empty(shape=len(precisions))
        nan_arr[:] = np.nan
        value_dict = {"precision": precisions,
                      "recall": recalls,
                      "dice": dices}
        value_dict.update({key: nan_arr for key in metrics})
        val_results = pd.DataFrame(value_dict,
                                   index=index).loc[:, col_order]  # ensure order
        # Transpose the results to have metrics in rows
        val_results = val_results.T
        # Add mean and set in first row
        means = metric_vals + [precisions.mean(), recalls.mean(), dices.mean()]
        val_results["mean"] = means
        cols = list(val_results.columns)
        cols.insert(0, cols.pop(cols.index('mean')))
        val_results = val_results.loc[:, cols]

        # Print the df to screen
        self.logger(highlighted(("[%s] Validation Results for "
                            "Epoch %i" % (name, epoch)).lstrip(" ")))
        print_string = val_results.round(self.print_round).to_string()
        self.logger(print_string.replace("NaN", "---") + "\n")

    def on_epoch_end(self, epoch, logs={}):
        self.logger("\n")
        # Predict and get CM
        TPs, relevant, selected, metrics = self.predict()
        for id_ in self.IDs:
            tp, rel, sel = TPs[id_], relevant[id_], selected[id_]
            precisions, recalls, dices = self._compute_dice(tp=tp, sel=sel, rel=rel)
            classes = np.arange(len(dices))

            # Add to log
            n = (id_ + "_") if len(self.IDs) > 1 else ""
            logs[f"{n}val_dice"] = dices.mean().round(self.log_round)
            logs[f"{n}val_precision"] = precisions.mean().round(self.log_round)
            logs[f"{n}val_recall"] = recalls.mean().round(self.log_round)
            for m_name, value in metrics[id_].items():
                logs[f"{n}val_{m_name}"] = value.round(self.log_round)

            if self.verbose:
                self._print_val_results(precisions=precisions,
                                        recalls=recalls,
                                        dices=dices,
                                        metrics=metrics[id_],
                                        epoch=epoch,
                                        name=id_,
                                        classes=classes)

        if len(self.IDs) > 1:
            # Print cross-dataset mean values
            if self.verbose:
                self.logger(highlighted(f"[ALL DATASETS] Means Across Classes"
                                        f" for Epoch {epoch}"))
            fetch = ("val_dice", "val_precision", "val_recall")
            m_fetch = tuple(["val_" + s for s in self.model.metrics_names])
            to_print = {}
            for f in fetch + m_fetch:
                scores = [logs["%s_%s" % (name, f)] for name in self.IDs]
                res = np.mean(scores)
                logs[f] = res.round(self.log_round)  # Add to log file
                to_print[f.split("_")[-1]] = list(scores) + [res]
            if self.verbose:
                df = pd.DataFrame(to_print)
                df.index = self.IDs + ["mean"]
                print(df.round(self.print_round))
            self.logger("")


class DividerLine(Callback):
    """
    Simply prints a line to screen after each epoch
    """
    def __init__(self, logger=None):
        """
        Args:
            logger: An instance of a MultiPlanar Logger that prints to screen
                    and/or file
        """
        super().__init__()
        self.logger = logger or ScreenLogger()

    def on_epoch_end(self, epoch, logs=None):
        self.logger("-"*45 + "\n")


class LearningCurve(Callback):
    """
    On epoch end this callback looks for all csv files matching the 'csv_regex'
    regex within the dir 'out_dir' and attempts to create a learning curve for
    each file that will be saved to 'out_dir'.
    Note: Failure to plot a learning curve based on a given csv file will
          is handled in the plot_all_training_curves function and will not
          cause the LearningCurve callback to raise an exception.
    """
    def __init__(self, log_dir="logs", out_dir="logs", fname="curve.png",
                 csv_regex="*training.csv", logger=None):
        """
        Args:
            log_dir: Relative path from the
            out_dir:
            fname:
            csv_regex:
            logger:
        """
        super().__init__()
        out_dir = os.path.abspath(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.csv_regex = os.path.join(os.path.abspath(log_dir), csv_regex)
        self.save_path = os.path.join(out_dir, fname)
        self.logger = logger or ScreenLogger()

    def on_epoch_end(self, epoch, logs={}):
        plot_all_training_curves(self.csv_regex,
                                 self.save_path,
                                 logy=True,
                                 raise_error=False,
                                 logger=self.logger)


class MemoryConsumption(Callback):
    def __init__(self, max_gib=None, round_=2, logger=None):
        self.max_gib = max_gib
        self.logger = logger
        self.round_ = round_

    def on_epoch_end(self, epoch, logs={}):
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_gib = round(mem_bytes / (1024**3), self.round_)
        logs['memory_usage_gib'] = mem_gib
        if self.max_gib and mem_gib >= self.max_gib:
            self.warn("Stopping training from callback 'MemoryConsumption'! "
                      "Total memory consumption of {} GiB exceeds limitation"
                      " (self.max_gib = {}) ".format(mem_gib, self.max_gib))
            self.model.stop_training = True


class DelayedCallback(object):
    """
    Callback wrapper that delays the functionality of another callback by N
    number of epochs.
    """
    def __init__(self, callback, start_from=0, logger=None):
        """
        Args:
            callback:   A tf.keras callback
            start_from: Delay the activity of 'callback' until this epoch
                        'start_from'
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        self.logger = logger or ScreenLogger()
        self.callback = callback
        self.start_from = start_from

    def __getattr__(self, item):
        return getattr(self.callback, item)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_from-1:
            self.callback.on_epoch_end(epoch, logs=logs)
        else:
            self.logger("[%s] Not active at epoch %i - will be at %i" %
                        (self.callback.__class__.__name__,
                         epoch+1, self.start_from))


class TrainTimer(Callback):
    """
    Appends train timing information to the log.
    If called prior to tf.keras.callbacks.CSVLogger this information will
    be written to disk.
    """
    def __init__(self, logger=None, max_minutes=None, verbose=1):
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.max_minutes = int(max_minutes) if max_minutes else None
        self.verbose = bool(verbose)

        # Timing attributes
        self.train_begin_time = None
        self.prev_epoch_time = None

    def on_train_begin(self, logs=None):
        self.train_begin_time = datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self.prev_epoch_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Compute epoch execution time
        end_time = datetime.now()
        epoch_time = end_time - self.prev_epoch_time
        train_time = end_time - self.train_begin_time

        # Update attributes
        self.prev_epoch_time = end_time

        # Add to logs
        train_hours = round(train_time.total_seconds() / 3600, 4)
        epoch_minutes = round(epoch_time.total_seconds() / 60, 4)
        logs["epoch_minutes"] = epoch_minutes
        logs["train_hours"] = train_hours

        if self.verbose:
            self.logger("[TrainTimer] Epoch time: %.2f minutes "
                        "- Total train time: %.2f hours"
                        % (epoch_minutes, train_hours))
        if self.max_minutes and train_hours*60 > self.max_minutes:
            self.logger("Stopping training. Training ran for {} minutes, "
                        "max_minutes of {} was specified on the TrainTimer "
                        "callback.".format(train_hours*60, self.max_minutes))
            self.model.stop_training = True


class FGBatchBalancer(Callback):
    """
    mpunet callback.
    Sets the forced FG fraction in a batch at each epoch to 1-recall over the
    validation data at the previous epoch
    """
    def __init__(self, train_data, val_data=None, logger=None):
        """
        Args:
            train_data: A mpunet.sequence object representing the
                        training data
            val_data:   A mpunet.sequence object representing the
                        validation data
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        super().__init__()
        self.data = (("train", train_data), ("val", val_data))
        self.logger = logger or ScreenLogger()
        self.active = True

    def on_epoch_end(self, epoch, logs=None):
        if not self.active:
            return None

        recall = logs.get("val_recall")
        if recall is None:
            self.logger("[FGBatchBalancer] No val_recall in logs. "
                        "Disabling callback. "
                        "Did you put this callback before the validation "
                        "callback?")
            self.active = False
        else:
            # Always at least 1 image slice
            fraction = max(0.01, 1 - recall)
            for name, data in self.data:
                if data is not None:
                    data.fg_batch_fraction = fraction
                    self.logger("[FGBatchBalancer] Setting FG fraction for %s "
                                "to: %.4f - Now %s/%s" % (name,
                                                          fraction,
                                                          data.n_fg_slices,
                                                          data.batch_size))


class MeanReduceLogArrays(Callback):
    """
    On epoch end, goes through the log and replaces any array entries with
    their mean value.
    """
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        for key, value in logs.items():
            if isinstance(value, (np.ndarray, list)):
                logs[key] = np.mean(value)


class PrintLayerWeights(Callback):
    """
    Print the weights of a specified layer every some epoch or batch.
    """
    def __init__(self, layer, every=10, first=10, per_epoch=False, logger=None):
        """
        Args:
            layer:      A tf.keras layer
            every:      Print the weights every 'every' batch or epoch if
                        per_epoch=True
            first:      Print the first 'first' elements of each weight matrix
            per_epoch:  Print after 'every' epoch instead of batch
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        super().__init__()
        if isinstance(layer, int):
            self.layer = self.model.layers[layer]
        else:
            self.layer = layer
        self.first = first
        self.every = every
        self.logger = logger or ScreenLogger()

        self.per_epoch = per_epoch
        if per_epoch:
            # Apply on every epoch instead of per batches
            self.on_epoch_begin = self.on_batch_begin
            self.on_batch_begin = lambda *args, **kwargs: None
        self.log()

    def log(self):
        self.logger("PrintLayerWeights Callback")
        self.logger("Layer:      ", self.layer)
        self.logger("Every:      ", self.every)
        self.logger("First:      ", self.first)
        self.logger("Per epoch:  ", self.per_epoch)

    def on_batch_begin(self, batch, logs=None):
        if batch % self.every:
            return
        weights = self.layer.get_weights()
        self.logger("Weights for layer '%s'" % self.layer)
        self.logger("Weights:\n%s" % weights[0].ravel()[:self.first])
        try:
            self.logger("Baises:\n%s" % weights[1].ravel()[:self.first])
        except IndexError:
            pass


class SaveOutputAs2DImage(Callback):
    """
    Save random 2D slices from the output of a given layer during training.
    """
    def __init__(self, layer, sequence, model, out_dir, every=10, logger=None):
        """
        Args:
            layer:    A tf.keras layer
            sequence: A MultiPlanar.sequence object from which batches are
                      sampled and pushed through the graph to output of layer
            model:    A tf.keras model object
            out_dir:  Path to directory (existing or non-existing) in which
                      images will be stored
            every:    Perform this operation every 'every' batches
        """
        super().__init__()
        self.every = every
        self.seq = sequence
        self.layer = layer
        self.epoch = None
        self.model = model
        self.logger = logger or ScreenLogger()

        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(self.out_dir)
        self.log()

    def log(self):
        self.logger("Save Output as 2D Image Callback")
        self.logger("Layer:      ", self.layer)
        self.logger("Every:      ", self.every)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if batch % self.every:
            return

        # Get output of layer
        self.model.predict_on_batch()
        sess = tf.keras.backend.get_session()
        X, _, _ = self.seq[0]
        outs = sess.run([self.layer.output], feed_dict={self.model.input: X})[0]
        if isinstance(outs, list):
            outs = outs[0]

        for i, (model_in, layer_out) in enumerate(zip(X, outs)):
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Plot model input and layer outputs on each ax
            chl1, axis, slice = imshow(ax1, model_in)
            chl2, _, _ = imshow(ax2, layer_out, axis=axis, slice=slice)

            # Set labels and save figure
            ax1.set_title("Model input - Channel %i - Axis %i - Slice %i"
                          % (chl1, axis,slice), size=22)
            ax2.set_title("Layer output - Channel %i - Axis %i - Slice %i"
                          % (chl2, axis, slice), size=22)

            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "epoch_%i_batch_%i_im_%i" %
                                     (self.epoch, batch, i)))
            plt.close(fig)


class SavePredictionImages(Callback):
    """
    Save images after each epoch of training of the model on a batch of
    training and a batch of validation data sampled from sequence objects.
    Saves the input image with ground truth overlay as well as the predicted
    label masks.
    """
    def __init__(self, train_data, val_data, outdir='images'):
        """
        Args:
            train_data: A mpunet.sequence object from which training
                        data can be sampled via the __getitem__ method.
            val_data:   A mpunet.sequence object from which validation
                        data can be sampled via the __getitem__ method.
            outdir:     Path to directory (existing or non-existing) in which
                        images will be stored.
        """
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.save_path = os.path.abspath(os.path.join(outdir, "pred_images_at_epoch"))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def pred_and_save(self, data, subdir):
        # Get a random batch
        X, y, _ = data[np.random.randint(len(data))]

        # Predict on the batch
        pred = self.model.predict(X)

        subdir = os.path.join(self.save_path, subdir)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # Plot each sample in the batch
        for i, (im, lab, p) in enumerate(zip(X, y, pred)):
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6))
            lab = lab.reshape(im.shape[:-1] + (lab.shape[-1],))
            p = p.reshape(im.shape[:-1] + (p.shape[-1],))
            # Imshow ground truth on ax2
            # This function will determine which channel, axis and slice to
            # show and return so that we can use them for the other 2 axes
            chnl, axis, slice = imshow_with_label_overlay(ax2, im, lab, lab_alpha=1.0)

            # Imshow pred on ax3
            imshow_with_label_overlay(ax3, im, p, lab_alpha=1.0,
                                      channel=chnl, axis=axis, slice=slice)

            # Imshow raw image on ax1
            # Chose the same slice, channel and axis as above
            im = im[..., chnl]
            im = np.moveaxis(im, axis, 0)
            if slice is not None:
                # Only for 3D imges
                im = im[slice]
            ax1.imshow(im, cmap="gray")

            # Set labels
            ax1.set_title("Image", size=18)
            ax2.set_title("True labels", size=18)
            ax3.set_title("Prediction", size=18)

            fig.tight_layout()
            with np.testing.suppress_warnings() as sup:
                sup.filter(UserWarning)
                fig.savefig(os.path.join(subdir, str(i) + ".png"))
            plt.close(fig.number)

    def on_epoch_end(self, epoch, logs={}):
        self.pred_and_save(self.train_data, "train_%s" % epoch)
        if self.val_data is not None:
            self.pred_and_save(self.val_data, "val_%s" % epoch)
            
