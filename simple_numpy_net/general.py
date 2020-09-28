# import standard modules

# import third party modules

# import third party modules


class Layer(object):
    """
    Parent class for all layer classes used for a model. For this framework is does only have one attribute
    which all layers share, however, it is important to keep things sorted in case of future enrichment.
    """

    def __init__(self):
        self.cache = None   # placeholder for incoming for modified data which must be captured
