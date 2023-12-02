import pyotp
import base64
import chardet

key = 'otpauth://totp/GitHub:IAKSH?secret=4WVJ227BXZAMUZOW&issuer=GitHub'
print(pyotp.TOTP(base64.b32encode(key).decode("utf-8")).now())
