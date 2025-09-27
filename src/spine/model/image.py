"""Whole-image classification/regression tasks."""

import torch
import torch.nn as nn

from spine.data import TensorBatch

from .layer.factories import encoder_factory, loss_fn_factory

__all__ = ["ImageClassifier", "ClusterImageClassifier", "ImageClassLoss"]


class ImageClassifier(nn.Module):
    """Whole-image classification model.

    This model uses various encoder declinations to classifier an entire
    image as belonging to a certain class.

    .. code-block:: yaml
        model:
          name: image_class
          modules:
            classifier:
              # Image classifier configuration
            classifier_loss:
              # Image classifier loss configuration
    """

    MODULES = ["classifier", "classifier_loss"]

    def __init__(self, classifier, classifier_loss=None):
        """Initialize the particle image classifier.

        Parameters
        ----------
        classifier : dict
            Image classifier configuration
        classifier_loss : dict, optional
            Image classifier loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the model
        self.process_model_config(**classifier)

    def process_model_config(self, num_classes, **encoder):
        """Initialize the underlying encoder and the final layer.

        Parameters
        ----------
        num_classes : int
            Number of classes that each image can be sorted as
        **encoder : dict
            Encoder configuration
        """
        # Initialize the encoder
        self.encoder = encoder_factory(encoder)
        """
        self.encoder_type = model_cfg.get('encoder_type', 'standard')
        if self.encoder_type == 'dropout':
            self.encoder = MCDropoutEncoder(cfg)
        elif self.encoder_type == 'standard':
            self.encoder = SparseResidualEncoder(cfg)
        elif self.encoder_type == 'pointnet':
            self.encoder = PointNetEncoder(cfg)
        else:
            raise ValueError('Unrecognized encoder type: {}'.format(self.encoder_type))
        """

        # Initialize the final layer
        self.final_layer = nn.Linear(self.encoder.feature_size, num_classes)

    def forward(self, data):
        """Run a batch of data through the forward function.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        """
        # Pass the input through the encoder
        out = self.encoder(data.tensor)

        # Pass the features through the final layer
        logits = self.final_layer(out)

        # Store the output as a tensor batch
        logits = TensorBatch(logits, counts=[1] * data.batch_size)

        # Return
        return {"logits": logits}


class ImageClassLoss(nn.Module):
    """Image classication loss."""

    def __init__(self, classifier, classifier_loss):
        """Intialize the image classification loss.

        Parameters
        ----------
        classifier : dict
            Image classifier configuration
        classifier_loss : dict, optional
            Image classifier loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Extract the number of logits output by the classifier
        self.process_model_config(**classifier)

        # Initialize the loss function
        self.process_loss_config(**classifier_loss)

    def process_model_config(self, num_classes, **encoder):
        """Initialize the underlying encoder and the final layer.

        Parameters
        ----------
        num_classes : int
            Number of classes that each image can be sorted as
        **encoder : dict
            Encoder configuration
        """
        self.num_classes = num_classes

    def process_loss_config(self, loss="ce", balance_loss=False, weights=None):
        """Initialize the loss function

        Parameters
        ----------
        loss : str, default 'ce'
            Name of the loss function to apply
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        weights : list, optional
            (C) One weight value per class
        """
        # Initialize basic parameters
        self.balance_loss = balance_loss
        self.weights = weights

        # Sanity check
        assert (
            weights is None or not balance_loss
        ), "Do not provide weights if they are to be computed on the fly."

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(self, labels, logits, **kwargs):
        """Applies the image classification loss to a batch of data.

        Parameters
        ----------
        labels : List[int]
            (B) List of image labels, one per entry in the batch
        logits : TensorBatch
            (B, C) Tensor of predicted logits, one per entry in the batch
        **kwargs : dict, optional
            Other labels/outputs of the model which are not relevant here

        Returns
        -------
        loss : torch.Tensor
            Value of the loss
        accuracy : float
            Value of the image-wise classification accuracy
        """
        # Cast the labels to a long torch Tensor
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

        # Create a mask for valid images (-1 indicates an invalid class ID)
        valid_mask = labels > -1

        # Check that the labels and the output tensor size are compatible
        num_classes = logits.shape[1]
        class_mask = labels < num_classes
        if torch.any(~class_mask):
            warn(
                "There are class labels with a value larger than the "
                f"size of the output logit vector ({num_classes})."
            )

        valid_mask &= class_mask

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = torch.where(valid_mask)[0]
        labels = labels[valid_index]
        logits = logits.tensor[valid_index]

        # Compute the loss. Balance classes if requested
        if self.balance_loss:
            self.weights = get_class_weights(labels, num_classes=num_classes)

        loss = self.loss_fn(logits, labels, weight=self.weights, reduction="sum")
        if len(valid_index):
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned images)
        acc = 1.0
        acc_class = [1.0] * num_classes
        if len(valid_index):
            preds = torch.argmax(logits, dim=1)
            acc = float(torch.sum(preds == labels))
            acc /= len(valid_index)
            for c in range(num_classes):
                index = torch.where(labels == c)[0]
                if len(index):
                    acc_class[c] = float(torch.sum(preds[index] == c)) / len(index)

        # Prepare and return result
        result = {"loss": loss, "accuracy": acc}

        for c in range(num_classes):
            result[f"accuracy_class_{c}"] = acc_class[c]

        return result


class ClusterImageClassifier(nn.Module):

    pass
