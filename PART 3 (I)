from pathlib import Path
import json,re
import joblib
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from typing import TypedDict
from langgraph.graph import StateGraph, END

ROOT=Path(__file__).resolve().parents[1]
class State(TypedDict, total=False):
    user_text:str; intent:str; order_id:int|None; retrieved:list; tool_result:dict|None; answer:dict; blocked:bool

SYSTEM_PROMPT='''ROLE: You are Flipkart support assistant. SPECIFIC: answer only the requested support intent. SHORT: keep answers concise. SURROUND: treat retrieved policy text and tool outputs as the only factual evidence. SINGLE: return exactly one JSON object with answer and source. Few-shot intent examples: "Can I return shoes?" -> policy; "Is order 12 risky?" -> return-risk; "What category is this image?" -> product-category.'''
INJECTION=["ignore previous instructions","ignore all rules","pretend you are"]

# Real Part 1 artifact.
risk_model=joblib.load(ROOT/"models/return_risk_model.pkl")
t_rf=json.loads((ROOT/"models/return_risk_threshold.json").read_text())["t_rf"]
# Real Part 2 artifact.
meta=json.loads((ROOT/"models/product_classifier_meta.json").read_text())
classes=meta["classes"]
clf=models.resnet18(weights=None); clf.fc=nn.Linear(512,10)
clf.load_state_dict(torch.load(ROOT/"models/product_classifier.pt",map_location="cpu")); clf.eval()
img_tf=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def check_return_risk(order_features:dict)->dict:
    import pandas as pd
    x=pd.DataFrame([order_features])
    p=float(risk_model.predict_proba(x)[0,1])
    return {"return_probability":p,"risk_bucket":"Low" if p<t_rf else ("High" if p>=t_rf+0.15 else "Medium"),"t_rf":t_rf}

def classify_product_image(image_path:str)->dict:
    x=img_tf(Image.open(image_path).convert("L")).unsqueeze(0)
    with torch.no_grad(): p=torch.softmax(clf(x),dim=1)[0]
    i=int(p.argmax()); return {"label":classes[i],"confidence":float(p[i])}

def intent_node(s:State):
    t=s["user_text"].lower()
    if any(x in t for x in INJECTION): return {"blocked":True,"intent":"blocked"}
    if re.search(r"\b(order|return risk|risky|risk)\b",t): return {"intent":"return-risk"}
    if re.search(r"\b(image|photo|picture|category|classify)\b",t) and re.search(r"\.png|\.jpg|image|photo|picture",t): return {"intent":"product-category"}
    return {"intent":"policy"}

def route(s:State): return "blocked" if s.get("blocked") else s["intent"]

def rag_node(s:State):
    from part3.rag_runtime import retrieve
    return {"retrieved":retrieve(s["user_text"],3)}

def tool_node(s:State):
    t=s["user_text"]
    if s["intent"]=="return-risk":
        m=re.search(r"order\s*#?\s*(\d+)",t,re.I)
        if not m: return {"tool_result":{"error":"Please provide an order ID and its features."}}
        oid=int(m.group(1)); import pandas as pd
        df=pd.read_csv(ROOT/"orders_dataset.csv"); row=df.loc[df.order_id==oid]
        if row.empty: return {"tool_result":{"error":"Order not found."}}
        rec=row.iloc[0].drop(labels=["returned"]).to_dict(); return {"order_id":oid,"tool_result":check_return_risk(rec)}
    m=re.search(r"(data/sample_images/[^\s]+\.png)",t)
    if m: return {"tool_result":classify_product_image(str(ROOT/m.group(1)))}
    return {"tool_result":{"error":"No image path supplied."}}

def response_node(s:State):
    if s.get("blocked"): return {"answer":{"answer":"I can’t follow instructions that attempt to override the support rules. Please ask a normal support question.","source":"policy kb"}}
    if s["intent"]=="policy":
        if not s.get("retrieved") or s["retrieved"][0]["score"]<0.35:
            score=s["retrieved"][0]["score"] if s.get("retrieved") else 0.0
            return {"answer":{"answer":f"I can’t answer that policy question from the available knowledge base. Best similarity score: {score:.3f}; refusal threshold: 0.350.","source":"policy kb"}}
        text=" ".join(x["text"] for x in s["retrieved"][:2])
        return {"answer":{"answer":text,"source":"policy kb"}}
    if s.get("tool_result") and "error" in s["tool_result"]: return {"answer":{"answer":s["tool_result"]["error"],"source":"return risk tool"}}
    src="return risk tool" if s["intent"]=="return-risk" else "image classifier tool"
    return {"answer":{"answer":json.dumps(s["tool_result"]),"source":src}}

def build_graph():
    g=StateGraph(State); g.add_node("intent",intent_node); g.add_node("rag",rag_node); g.add_node("tool",tool_node); g.add_node("response",response_node)
    g.set_entry_point("intent"); g.add_conditional_edges("intent",route,{"policy":"rag","return-risk":"tool","product-category":"tool","blocked":"response"})
    g.add_edge("rag","response"); g.add_edge("tool","response"); g.add_edge("response",END); return g.compile()
