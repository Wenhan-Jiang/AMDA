# Optimizer AMDA

from AMDA import AMDA

# vanilla AMDA
optimizer = AMDA(net.parameters(), lr=1e-3, weight_decay=5e-4)

# When you are using AdamW as a default optimizer, you should set weight_decouple=True
optimizer = AMDA(net.parameters(), lr=1e-3, weight_decay=0.01, weight_decouple=True) 
