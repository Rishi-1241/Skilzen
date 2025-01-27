import base64
chunk = base64.b64decode("GAAIAAgACAAIAAgACAAIAAgACAD4/wgA+P8IAPj/+P/o//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4/+j/+P/4//j/+P/4/wgA+P/4//j/+P/4//j/CAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAYAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAD4//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4/+j/+P/4//j/+P/4//j/+P/4/wgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgA+P8=")
print(chunk)

import binascii

# Base64 encoded data
base64_data = "GAAIAAgACAAIAAgACAAIAAgACAD4/wgA+P8IAPj/+P/o//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4/+j/+P/4//j/+P/4/wgA+P/4//j/+P/4//j/CAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAYAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAD4//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4//j/+P/4/+j/+P/4//j/+P/4//j/+P/4/wgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgA+P8="

# Decode the data
decoded_data = binascii.a2b_base64(base64_data)
print(decoded_data)

import pybase64

# Base64 encoded data

# Decode the data
ans = pybase64.b64decode(base64_data)
print(ans)

