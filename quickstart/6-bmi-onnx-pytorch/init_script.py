import onnxruntime as ort

from utils import predict, BBox
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device {}".format(device))

face_detector = ort.InferenceSession('models/version-RFB-320.onnx')
face_detector_input = face_detector.get_inputs()[0].name
model_path = r"experiments/simple_experiment/model.pt"
model = torch.jit.load(model_path)

if torch.cuda.is_available():
    model.cuda()
