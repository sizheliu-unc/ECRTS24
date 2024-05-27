This folder stores DL models. Please note that we do not claim authorship of the .onnx file nor the model architectures located in this folder. Their references can be located in the publication.

.onnx files are original DL models in ONNX format.

.engine files are TensorRT engine plans (i.e. inference runtime engine). This file is tied to the
specific hardware, NVIDIA driver, TensorRT, CUDA Runtime, cuDNN, etc. used when generating the engine plan.

.info.log files are TensorRT logs generated when converting the model from onnx.

Only .engine files are used in performing our experiments.

ViT-S (224x224):
- vit_small_patch16_224_Opset16.onnx
- vit.engine
- vit.info.log

RegNet-Y (224x224):
- regnet_y_16gf_Opset16.onnx
- regnet.engine
- regnet.info.log

DeiT-B (224x224):
- deit3_base_patch16_224_Opset17.onnx
- deit.engine
- deit.info.log

DETR-R50 (512x512):
- detr.onnx
- detr.engine
- detr.info.log

SegFormer-B1 (512x512):
- segformer.b1.512x512.ade.160k.onnx
- segformer.engine
- segformer.info.log


**If the user wishes to regenerate the engine plan files, do the following:**

```
trtexec --onnx={model onnx file} --saveEngine={model engine file} --dumpLayerInfo --dumpProfile --separateProfileRun &> {model_name}.info.log
```

For example:
```
trtexec --onnx=detr.onnx --saveEngine=detr.engine --dumpLayerInfo &> detr.info.log
```
Will generate the engine plan for DETR-R50 and dump the model's timing information to detr.info.log.

Alternatively, run:
```bash
alias trtexec={TensorRT folder}/bin/trtexec
./generate_engine_plans.sh
```
will generate the engine plan for all DL models located in this folder.
 