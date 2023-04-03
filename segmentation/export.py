import io
import numpy as np

from torch import nn
import torch.onnx

def export_model(model_path, device):
    model = torch.load(f=model_path, map_location=device)
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    input_names = [ "actual_input" ]
    output_names = [ "output" ]
    dynamic_axes_dict = {
        'actual_input': {
            0: 'bs',
            2: 'img_x',
        3: 'img_y'
        },
        'Output': {
            0: 'bs'
        }
    } 

 
    torch.onnx.export(model,
                 dummy_input,
                 "segmentation/unet16_model.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 dynamic_axes=dynamic_axes_dict,
                 export_params=True,
                 )

    # torch.onnx.export(model,
    #              dummy_input,
    #              "segmentation/unet16_model.onnx",
    #              verbose=False,
    #              input_names=input_names,
    #              output_names=output_names,
    #              export_params=True,
    #              )
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    export_model("segmentation/unet16_model.pt", device)

