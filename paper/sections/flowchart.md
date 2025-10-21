---
config:
  theme: default
  layout: elk
  look: classic
---
flowchart TB
    A["<b>Dataset<br></b>"] --> S1["<b>Stratified Data Preparation<br></b>"]
    S1 --> S2["<b>Synthetic Generation<br></b>"]
    S2 --> S3["<b>Systematic Mixing and Classification<br></b>"]
    S3 --> S4["<b>Evaluation<br></b>"]
     A:::dataStyle
     S1:::stage1Style
     S2:::stage2Style
     S3:::stage3Style
     S4:::stage4Style
    classDef dataStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#000
    classDef stage1Style fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#000
    classDef stage2Style fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#000
    classDef stage3Style fill:#FFF9C4,stroke:#F9A825,stroke-width:2px,color:#000
    classDef stage4Style fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
