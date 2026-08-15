from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; MODEL_DIR = ROOT / "models"; SAMPLE_DIR = DATA / "sample_images"
MODEL_DIR.mkdir(exist_ok=True); SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
base_train = datasets.FashionMNIST(DATA / "fashion_mnist", train=True, download=True)
base_test = datasets.FashionMNIST(DATA / "fashion_mnist", train=False, download=True)
labels = np.array(base_train.targets)
tr_idx, va_idx = train_test_split(np.arange(len(base_train)), test_size=5000, stratify=labels, random_state=SEED)
train_ds = Subset(datasets.FashionMNIST(DATA / "fashion_mnist", train=True, transform=train_tf), tr_idx)
val_ds = Subset(datasets.FashionMNIST(DATA / "fashion_mnist", train=True, transform=train_tf), va_idx)
test_ds = datasets.FashionMNIST(DATA / "fashion_mnist", train=False, transform=train_tf)

# Feature extractor: pretrained ResNet-18, early/middle layers frozen.
weights = models.ResNet18_Weights.DEFAULT
backbone = models.resnet18(weights=weights)
backbone.fc = nn.Identity()
for p in backbone.parameters(): p.requires_grad = False
backbone.eval()

def cache_features(ds, batch_size=128):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, ys = [], []
    with torch.no_grad():
        for x,y in loader:
            feats.append(backbone(x)); ys.append(y)
    return torch.cat(feats), torch.cat(ys)

# CPU-safe default; set CUDA_VISIBLE_DEVICES or edit device for GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)

def cache_features_device(ds, batch_size=128):
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    fs=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            fs.append(backbone(x.to(device)).cpu()); ys.append(y)
    return torch.cat(fs), torch.cat(ys)

Xtr,ytr=cache_features_device(train_ds); Xv,yv=cache_features_device(val_ds)
head=nn.Linear(512,10).to(device)
opt=torch.optim.Adam(head.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()
for _ in range(10):
    head.train()
    perm=torch.randperm(len(Xtr))
    for start in range(0,len(Xtr),256):
        ix=perm[start:start+256]; xb=Xtr[ix].to(device); yb=ytr[ix].to(device)
        opt.zero_grad(); loss=loss_fn(head(xb),yb); loss.backward(); opt.step()

def acc_features(X,y):
    head.eval()
    with torch.no_grad(): pred=head(X.to(device)).argmax(1).cpu().numpy()
    return accuracy_score(y.numpy(),pred)
val_before=acc_features(Xv,yv)

# If <80%, unfreeze layer4 and fine-tune the full model with a smaller LR.
model=backbone
model.fc=head
if val_before < 0.80:
    for p in model.layer4.parameters(): p.requires_grad=True
    for p in model.fc.parameters(): p.requires_grad=True
    model.to(device)
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),lr=1e-4)
    loader=DataLoader(train_ds,batch_size=128,shuffle=True,num_workers=0)
    model.train()
    for _ in range(3):
        for x,y in loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
else:
    model.to(device)
val_after=val_before if val_before>=0.80 else None

# Test evaluation
loader=DataLoader(test_ds,batch_size=128,shuffle=False,num_workers=0)
model.eval(); preds=[]; ys=[]
with torch.no_grad():
    for x,y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy()); ys.extend(y.numpy())
acc=accuracy_score(ys,preds); cm=confusion_matrix(ys,preds)
report=classification_report(ys,preds,target_names=classes,output_dict=True,zero_division=0)

# Save actual PNG samples for Part 3.
raw_test=datasets.FashionMNIST(DATA / "fashion_mnist", train=False, download=False)
for i in [0,1,2,3,4]: raw_test[i][0].save(SAMPLE_DIR/f"sample_{i}.png")

torch.save(model.state_dict(), MODEL_DIR/"product_classifier.pt")
(MODEL_DIR/"product_classifier_meta.json").write_text(json.dumps({"classes":classes,"input_size":224,"val_before":val_before,"val_after":val_after},indent=2))
np.savetxt(ROOT/"outputs/confusion_matrix.csv",cm,fmt="%d",delimiter=",")
(ROOT/"outputs/part2_report.json").write_text(json.dumps({"test_accuracy":acc,"confusion_matrix":cm.tolist(),"classification_report":report,"val_before":val_before,"val_after":val_after},indent=2))
print("test accuracy",acc); print(cm)
