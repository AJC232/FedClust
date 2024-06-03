def update_global_model(global_model, local_models, data_sizes):
    global_w = global_model.state_dict()

    # Calculate total size of all datasets
    total_size = sum(data_sizes)

    fed_avg_freqs = [data_size / total_size for data_size in data_sizes]
    print(fed_avg_freqs)

    i = 0
    for model in local_models:
      model_para = model.state_dict()
      if i == 0:
        for key in model_para:
          global_w[key] = model_para[key] * fed_avg_freqs[i]
      else:
        for key in model_para:
          global_w[key] += model_para[key] * fed_avg_freqs[i]
      i+=1

    # # Update the global model
    global_model.load_state_dict(global_w)

    return global_model
