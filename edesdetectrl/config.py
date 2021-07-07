import toml


def merge(source, destination):
    """
    Deeply merge two dictionaries, overwriting values in destination with values of the same keys in source.
    Copied from https://stackoverflow.com/questions/20656135/python-deep-merge-dictionary-data :)
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


config_upstream = toml.load("config_upstream.toml")  # Config that is included in git
config_local = toml.load("config_local.toml")        # Config that has to be set locally by the dev
config = merge(config_local, config_upstream)        # Config, combined!

# from edesdetectrl.config import config
