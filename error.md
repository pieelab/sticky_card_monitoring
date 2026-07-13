C:\Users\ALANalysis\sticky_card_monitoring\src\test_crop_labeller.py:191: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  stage1_model = torch.load(stage1_model_path)
Traceback (most recent call last):
  File "C:\Users\ALANalysis\sticky_card_monitoring\src\test_crop_labeller.py", line 466, in <module>
    main()
  File "C:\Users\ALANalysis\sticky_card_monitoring\src\test_crop_labeller.py", line 463, in main
    classify_prep(**cli_args())
  File "C:\Users\ALANalysis\sticky_card_monitoring\src\test_crop_labeller.py", line 428, in classify_prep
    destination_dir = classify_segments_hierarchical(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ALANalysis\sticky_card_monitoring\src\test_crop_labeller.py", line 191, in classify_segments_hierarchical
    stage1_model = torch.load(stage1_model_path)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ALANalysis\.conda\envs\sticky-card-classifier\Lib\site-packages\torch\serialization.py", line 1360, in load
    return _load(
           ^^^^^^
  File "C:\Users\ALANalysis\.conda\envs\sticky-card-classifier\Lib\site-packages\torch\serialization.py", line 1848, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "C:\Users\ALANalysis\.conda\envs\sticky-card-classifier\Lib\site-packages\torch\serialization.py", line 1837, in find_class
    return super().find_class(mod_name, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'dinov2'