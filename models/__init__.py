def get_network(network_name):
    network_name = network_name.lower()

    if network_name == 'gadanext':
        from .GAdaNext import GAdaNext
        return GAdaNext
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))