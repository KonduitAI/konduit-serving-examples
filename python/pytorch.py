from PIL import Image 
import torchvision.transforms as transforms
import onnxruntime
import os 

dl_path = os.path.abspath("../data/facedetector/facedetector.onnx")

image = Image.fromarray(image.astype('uint8')[0], 'RGB')
resize = transforms.Resize([240, 320])
img_y = resize(image)
to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession(dl_path)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]