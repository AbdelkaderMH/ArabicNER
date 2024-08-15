import os
import logging
import torch
import numpy as np
from arabiner.trainers import BaseTrainer
from arabiner.utils.metrics import compute_nested_metrics, compute_single_label_metrics
from arabiner.utils import loss_functions
from arabiner.utils.misc import Nonparametric
logger = logging.getLogger(__name__)


class BertMTNTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.criterion = loss_functions.FocalTverskyLoss(weight_focal=0.2, weight_ce=0.8, gamma=2)
        self.criterion = loss_functions.UnifiedLoss()

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)
        num_labels = [len(v) for v in self.train_dataloader.dataset.vocab.tags[1:]]
        patience = self.patience

        for epoch_index in range(self.max_epochs):
            self.current_epoch = epoch_index
            train_loss = 0

            for batch_index, (subwords, gold_tags, tokens, valid_len, logits) in enumerate(self.tag(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                
                #print('logits:', logits.shape)
                #print('labels:', gold_tags.shape)
                #print('one y_label', torch.reshape(gold_tags[:, 1, :], (-1,)).long().shape)
                #print('logits_one_labels', logits[:, :, 1, 0:4].view(-1, logits[:, :, 1, 0:4].shape[-1]).shape)
             

                # Compute loses for each output
                # logits = B x T x L x C
            
                losses = [self.criterion(logits[:, :, i, 0:4].view(-1, logits[:, :, i, 0:4].shape[-1]),
                                    torch.reshape(gold_tags[:, i, :], (-1,)).long())
                          for i, l in enumerate(num_labels)]

                all_losses = torch.stack(losses)               
                mean = all_losses.mean()
                penalty = 5 * ((all_losses - mean) ** 2).mean()
                #losses.append(penalty)
    
                #losses.append(loss_functions.Entropy(logits))
            
                torch.autograd.backward(losses)
                # Avoid exploding gradient by doing gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                self.scheduler.step()
                batch_loss = sum(l.item() for l in losses)
                train_loss += batch_loss

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        batch_loss
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")
            val_preds, segments, valid_len, val_loss = self.eval(self.val_dataloader)
            val_metrics = compute_single_label_metrics(segments)#, self.val_dataloader.dataset.transform.vocab.tags[1:])

            epoch_summary_loss = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            epoch_summary_metrics = {
                "val_micro_f1": val_metrics.micro_f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall
            }

            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 %f",
                epoch_index,
                self.current_timestep,
                train_loss,
                val_loss,
                val_metrics.micro_f1
            )

            if val_loss < best_val_loss:
                patience = self.patience
                best_val_loss = val_loss
                logger.info("** Validation improved, evaluating test data **")
                test_preds, segments, valid_len, test_loss = self.eval(self.test_dataloader)
                self.segments_to_file(segments, os.path.join(self.output_path, "predictions.txt"))
                test_metrics = compute_single_label_metrics(segments)#, self.test_dataloader.dataset.transform.vocab.tags[1:])

                epoch_summary_loss["test_loss"] = test_loss
                epoch_summary_metrics["test_micro_f1"] = test_metrics.micro_f1
                epoch_summary_metrics["test_precision"] = test_metrics.precision
                epoch_summary_metrics["test_recall"] = test_metrics.recall

                logger.info(
                    f"Epoch %d | Timestep %d | Test Loss %f | F1 %f",
                    epoch_index,
                    self.current_timestep,
                    test_loss,
                    test_metrics.micro_f1
                )

                self.save()
            else:
                patience -= 1

            # No improvements, terminating early
            if patience == 0:
                logger.info("Early termination triggered")
                break

            self.summary_writer.add_scalars("Loss", epoch_summary_loss, global_step=self.current_timestep)
            self.summary_writer.add_scalars("Metrics", epoch_summary_metrics, global_step=self.current_timestep)

    def tag(self, dataloader, is_train=True):
        """
        Given a dataloader containing segments, predict the tags
        :param dataloader: torch.utils.data.DataLoader
        :param is_train: boolean - True for training model, False for evaluation
        :return: Iterator
                    subwords (B x T x NUM_LABELS)- torch.Tensor - BERT subword ID
                    gold_tags (B x T x NUM_LABELS) - torch.Tensor - ground truth tags IDs
                    tokens - List[arabiner.data.dataset.Token] - list of tokens
                    valid_len (B x 1) - int - valiud length of each sequence
                    logits (B x T x NUM_LABELS) - logits for each token and each tag
        """
        for subwords, gold_tags, tokens, mask, valid_len in dataloader:
            self.model.train(is_train)

            if torch.cuda.is_available():
                subwords = subwords.cuda()
                gold_tags = gold_tags.cuda()

            if is_train:
                self.optimizer.zero_grad()
                logits = self.model(subwords)
            else:
                with torch.no_grad():
                    logits = self.model(subwords)

            yield subwords, gold_tags, tokens, valid_len, logits

    def eval(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()
        num_labels = [len(v) for v in dataloader.dataset.vocab.tags[1:]]
        loss = 0

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            losses = [self.criterion(logits[:, :, i, 0:4].view(-1, logits[:, :, i, 0:4].shape[-1]),
                                    torch.reshape(gold_tags[:, i, :], (-1,)).long())
                          for i, l in enumerate(num_labels)]
            all_losses = torch.stack(losses)               
            mean = all_losses.mean()
            penalty = 5 * ((all_losses - mean) ** 2).mean()
            #losses.append(penalty)

 
            loss += sum(losses)
            preds += torch.argmax(logits, dim=3)
            segments += tokens
            valid_lens += list(valid_len)

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)

        return preds, segments, valid_lens, loss

    def infer(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            preds += torch.argmax(logits, dim=3)
            segments += tokens
            valid_lens += list(valid_len)

        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)
        return segments
    def keep_one_non_o_tag(self, data):
        """
        Keep one non-'O' tag or 'O' if all values are 'O'.

        Parameters:
        - data (list of dict): List of dictionaries, each containing a 'tag' key with a value.

        Returns:
        - list: List containing a single dictionary with the selected 'tag' key and value.
        """
        non_o_tags = [item['tag'] for item in data if (item['tag'] != "O" and 'pad' not in item['tag'])]

        if len(non_o_tags)==0 :
            # If all values are 'O', keep 'O'
            return [{'tag': "O"}]
        else:
            # Keep the first non-'O' value
            return [{'tag': non_o_tags[0]}]
        
    def to_segments(self, segments, preds, valid_lens, vocab):
        if vocab is None:
            vocab = self.vocab

        tagged_segments = list()
        tokens_stoi = vocab.tokens.get_stoi()
        unk_id = tokens_stoi["UNK"]

        for segment, pred, valid_len in zip(segments, preds, valid_lens):
            # First, the token at 0th index [CLS] and token at nth index [SEP]
            # Combine the tokens with their corresponding predictions
            segment_pred = zip(segment[1:valid_len-1], pred[1:valid_len-1])

            # Ignore the sub-tokens/subwords, which are identified with text being UNK
            segment_pred = list(filter(lambda t: tokens_stoi[t[0].text] != unk_id, segment_pred))

            # Attach the predicted tags to each token
            list(map(lambda t: setattr(t[0], 'pred_tag', self.keep_one_non_o_tag([{"tag": vocab.get_itos()[tag_id]}
                                                     for tag_id, vocab in zip(t[1].int().tolist(), vocab.tags[1:])])), segment_pred))
            
            # We are only interested in the tagged tokens, we do no longer need raw model predictions
            tagged_segment = [t for t, _ in segment_pred]
            #print(tagged_segment)
            tagged_segments.append(tagged_segment)

        return tagged_segments
