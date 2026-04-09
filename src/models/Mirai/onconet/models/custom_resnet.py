from torch import nn

from .factory import RegisterModel, load_pretrained_weights, get_layers
from .default_resnets import load_pretrained_model
from .resnet_base import ResNet



def validate_raw_block_layout(raw_block_layout):
    """Confirms that a raw block layout is in the right format.

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout in the format
            'block_name,num_repeats-block_name,num_repeats-...'

    Raises:
        Exception if the raw block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. block_name,num_repeats)
    for raw_layer_layout in raw_block_layout:
        for raw_block_spec in raw_layer_layout.split('-'):
            if len(raw_block_spec.split(',')) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(raw_block_spec))


def parse_block_layout(raw_block_layout):
    """Parses a ResNet block layout, which is a list of layer layouts
    with each layer layout in the form 'block_name,num_repeats-block_name,num_repeats-...'

    Example:
        ['BasicBlock,2',
         'BasicBlock,1-NonLocalBlock,1',
         'BasicBlock,3-NonLocalBlock,2-Bottleneck,2',
         'BasicBlock,2']
        ==>
        [[('BasicBlock', 2)],
         [('BasicBlock', 1), ('NonLocalBlock', 1)],
         [('BasicBlock', 3), ('NonLocalBlock', 2), ('Bottleneck', 2)],
         [('BasicBlock', 2)]]

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout as described above.

    Returns:
        A list of lists of length 4 (one for each layer of ResNet). Each inner list is
        a list of tuples, where each tuple is (block_name, num_repeats).
    """

    validate_raw_block_layout(raw_block_layout)

    block_layout = []
    for raw_layer_layout in raw_block_layout:
        raw_block_specs = raw_layer_layout.split('-')
        layer = [raw_block_spec.split(',') for raw_block_spec in raw_block_specs]
        layer = [(block_name, int(num_repeats)) for block_name, num_repeats in layer]
        block_layout.append(layer)

    return block_layout


@RegisterModel("custom_resnet")
class CustomResnet(nn.Module):
    def __init__(self, args):
        super(CustomResnet, self).__init__()
        nested_block_layout = parse_block_layout(args.block_layout)

        layers = get_layers(nested_block_layout)
        self._model = ResNet(layers, args)
        model_name = args.pretrained_imagenet_model_name
        if args.pretrained_on_imagenet:
            load_pretrained_weights(self._model,
                                    load_pretrained_model(model_name))

    def forward(self, x, risk_factors=None, batch=None):
        return self._model(x, risk_factors=risk_factors, batch=None)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self
