from pathlib import Path
from part3.rag_runtime import retrieve
queries=[
("What is the return window for apparel?",{"return_apparel"}),
("How long can I return an electronics item?",{"return_electronics"}),
("When is a refund initiated after a return?",{"refund"}),
("How long does standard delivery take?",{"delivery_standard"}),
("Can a courier collect my return from my pin code?",{"reverse_pickup"}),
]
ps=[]; rs=[]
for q,answer in queries:
    docs=[]
    for x in retrieve(q,3):
        if x["doc_id"] not in docs: docs.append(x["doc_id"])
    hit=sum(d in answer for d in docs); p=hit/3; r=hit/len(answer); ps.append(p); rs.append(r)
    print(f"{q}\nPrecision@3 = {hit}/3 = {p:.3f}; Recall@3 = {hit}/{len(answer)} = {r:.3f}; docs={docs}\n")
print("Average Precision@3",sum(ps)/len(ps)); print("Average Recall@3",sum(rs)/len(rs))
