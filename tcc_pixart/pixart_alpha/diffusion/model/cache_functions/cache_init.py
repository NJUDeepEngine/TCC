def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}

    for j in range(28):
        cache[-1][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1][j] = {}
        cache_dic['cross_attn_map'][-1][j] = {}

    cache_dic['cache_type'] = model_kwargs.get('cache_type', 'attention')
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = model_kwargs.get('ratio_scheduler', 'ToCa')
    cache_dic['fresh_ratio'] = model_kwargs.get('fresh_ratio', 0.30)
    cache_dic['fresh_threshold'] = model_kwargs.get('fresh_threshold', 3)
    cache_dic['force_fresh'] = model_kwargs.get('force_fresh', 'global')
    cache_dic['soft_fresh_weight'] = model_kwargs.get('soft_fresh_weight', 0.25)
    cache_dic['use_toca'] = bool(model_kwargs.get('use_toca', True))
    cache_dic['strict_force_fresh'] = bool(model_kwargs.get('strict_force_fresh', False))
    cache_dic['cache_mode'] = model_kwargs.get('cache_mode', 'toca')
    cache_dic['tcc_collector'] = model_kwargs.get('tcc_collector', None)
    cache_dic['tcc_corrector'] = model_kwargs.get('tcc_corrector', None)
    cache_dic['tcc_force_full'] = bool(model_kwargs.get('tcc_force_full', False))
    cache_dic['test_FLOPs'] = bool(model_kwargs.get('test_FLOPs', False))
    cache_dic['flops'] = 0.0
    cache_dic['model_flops'] = 0.0
    cache_dic['tcc_flops'] = 0.0
    cache_dic['profile'] = model_kwargs.get('profile', None)
    current = {}
    current['num_steps'] = num_steps
    return cache_dic, current
    
