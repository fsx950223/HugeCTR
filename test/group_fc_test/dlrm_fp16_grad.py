import hugectr
from mpi4py import MPI
# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(model_name = "dlrm",
                              max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True,
                              use_mixed_precision = True,
                              scaler = 1024)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./dcn_data/file_list.txt"],
                                  eval_source = "./dcn_data/file_list_test.txt",
                                  check_type = hugectr.Check_t.Sum)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0001)
# 2. Initialize the Model instance
model = hugectr.Model(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 3000,
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["dense"],
                            top_name_list = ["fc1", "fc2", "fc3"],
                            num_outputs = [512, 256, 128],
                            last_act_type = hugectr.Activation_t.Relu))                   
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["fc3","sparse_embedding1"],
                            top_names = ["interaction1", "interaction1_grad"]))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["interaction1", "interaction1_grad"],
                            top_name_list = ["fc4", "fc5", "fc6", "fc7", "fc8"],
                            num_outputs = [1024, 1024, 512, 256, 1],
                            last_act_type = hugectr.Activation_t.Non))                                                                             
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
# 4. Dump the Model graph to JSON
# model.graph_to_json(graph_config_file = "dlrm.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 200000, snapshot_prefix = "dlrm")