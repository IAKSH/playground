package com.company;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.io.File;

public class Main {

    public static boolean isRunning = false;

    private static String orSettings = "<Times>10</Times>";

    public static int times;

    //此窗口会托管给另一个线程运行，不受后台进程干扰
    public static Window mainWindow = new Window("(*≥▽≤)-ZR",250,200);

    public static void main(String[] args){
        //读取设置
        checkSettings();

        //一直等待执行
        while(true){

            if(isRunning){

                mainWindow.startButton.setEnabled(false);
                mainWindow.startButton.setText("正在运行");

                if(mainWindow.inputArea.getText() == ""){
                    System.out.println("错误：没有输入文本");
                }else{
                    try{
                        runTranslate();
                    }catch (InterruptedException e){
                        e.printStackTrace();
                    }
                }

                isRunning = false;
                mainWindow.startButton.setEnabled(true);
                mainWindow.startButton.setText("开始");
            }

            try{
                Thread.sleep(1000);
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }

    private static void runTranslate() throws InterruptedException {
        Translater t = new Translater();
        Translater2 t1 = new Translater2();

        String orAboutLabelText = mainWindow.aboutLabel.getText();

        System.out.println("------------- 开始翻译 -------------");
        String strRes = t.Translate(mainWindow.inputArea.getText());
        strRes = StringSub.subString(JSONGet.loadJson(strRes),"\"dst\":\"","\"}]}");
        strRes = StringSub.unicodeToCn(strRes);
        System.out.println("第0次翻译(中文结果)" + strRes);
        Thread.sleep(1000);
        //循环翻译
        for(int i = 0;i<times;i++){

            mainWindow.aboutLabel.setText("当前进度：" + i + '/' + times);

            strRes = StringSub.subString(JSONGet.loadJson(t1.Translate(strRes)),",\"dst\":\"","\"}]}");
            System.out.println('第' + String.valueOf(i+1) + "次翻译(英文结果)" + strRes);
            Thread.sleep(1000);

            strRes = StringSub.subString(JSONGet.loadJson(t.Translate(strRes)),",\"dst\":\"","\"}]}");
            strRes = StringSub.unicodeToCn(strRes);
            System.out.println('第' + String.valueOf(i+1) + "次翻译(中文结果)" + strRes);
            Thread.sleep(1000);
        }
        mainWindow.aboutLabel.setText(orAboutLabelText);
        System.out.println("------------- 翻译结束 -------------");
        //输出结果到剪切板，然后显示提示对话框
        setSysClipboardText(strRes);
        JOptionPane.showMessageDialog(null, "翻译结果:" + '"' + strRes + '"' + "\n\r已复制到系统剪切板,按ctrl+v粘贴。");
    }

    //复制字符串到剪切板
    public static void setSysClipboardText(String writeMe) {
        Clipboard clip = Toolkit.getDefaultToolkit().getSystemClipboard();
        Transferable tText = new StringSelection(writeMe);
        clip.setContents(tText, null);
    }

    public static void checkSettings(){
        File settings = new File("settings.txt");
        if(!settings.exists()){
            System.out.print("main: 未找到设置文件");

            TxtRW.writeTxt("settings.txt",orSettings);
            times = 10;

            System.out.println("，已自动创建");
        }else{
            System.out.print("main: 已找到设置文件，正在读取. . .");
            String temp = TxtRW.readTxt("settings.txt");
            times = Integer.parseInt(StringSub.subString(temp,"<Times>","</Times>"));
            System.out.println("完成！");
        }
    }
}
