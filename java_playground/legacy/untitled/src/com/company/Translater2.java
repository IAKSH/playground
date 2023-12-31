package com.company;

public class Translater2 extends Translater{
    public String TRANS_FROM = "zh";//翻译源语言,初始值中文
    public String TRANS_TO = "en";//译文语言,初始值英文

    public String Translate(String input){

        //获取随机数
        double d = Math.random();
        TRANS_SALT = (int) (d * 100);

        //拼接字符串
        String str1 = TRANS_APPID + input + TRANS_SALT + TRANS_KEY;

        //System.out.println(str1);
        //获得签名（md5）
        TRANS_SIGN = Encryption.stringToMD5(str1);


        //拼接完整请求
        String TRANS_FINAL = "http://api.fanyi.baidu.com/api/trans/vip/translate?q=" + StringSub.getURLEncoderString(input) + '&' +"from=" + TRANS_FROM +'&' + "to=" + TRANS_TO + '&' + "appid=" + TRANS_APPID + '&' + "salt=" + TRANS_SALT + '&' + "sign=" + TRANS_SIGN;
        //System.out.println(TRANS_FINAL);
        return TRANS_FINAL;
    }
}
