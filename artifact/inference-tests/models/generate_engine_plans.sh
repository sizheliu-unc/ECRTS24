#!/bin/bash
trtexec --onnx=deit.onnx --saveEngine=deit.engine --dumpLayerInfo &> deit.info.log
trtexec --onnx=detr.onnx --saveEngine=detr.engine --dumpLayerInfo &> detr.info.log
trtexec --onnx=segformer.onnx --saveEngine=segformer.engine --dumpLayerInfo &> segformer.info.log
trtexec --onnx=regnet.onnx --saveEngine=regnet.engine --dumpLayerInfo &> regnet.info.log
trtexec --onnx=vit.onnx --saveEngine=vit.engine --dumpLayerInfo &> vit.info.log
