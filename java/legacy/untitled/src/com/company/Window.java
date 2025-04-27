package com.company;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

public class Window extends JFrame {
    public Window(String Title,int w,int h){
        //配置窗体属性
        System.out.print("mainWindow: 正在加载窗体. . .");
        setTitle(Title);
        setSize(w,h);
        setMinimumSize(new Dimension(w,h));
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        this.setIconImage(new ImageIcon("icon.png").getImage());

        setVisible(true);
        System.out.println("完成！");

        //加载布局
        System.out.print("mainWindow: 正在加载布局. . .");
        loadPanel();
        System.out.println("完成！");


        //加载控件
        System.out.print("mainWindow: 正在加载控件. . .");
        loadComp();
        System.out.println("完成！");
    }

    //声明控件
    public JPanel panel = new JPanel();
    public JButton startButton = new JButton();
    public JLabel aboutLabel = new JLabel();
    public JTextArea inputArea = new JTextArea();

    //声明其他东西

    private void loadPanel(){
        panel.setLayout(new BorderLayout());
        this.add(panel);
    }
    private void loadComp(){
        //修改控件外观
        aboutLabel.setText("输入原文，点击翻译。（注：不要换行）");
        inputArea.setText("这是一串示例字符，仅此而已。");
        startButton.setText("开始");
        inputArea.setBackground(Color.white);
        startButton.setBackground(Color.lightGray);

        //添加按钮监听
        startButton.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Main.isRunning=true;
            }

            @Override
            public void mousePressed(MouseEvent e) {

            }

            @Override
            public void mouseReleased(MouseEvent e) {

            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }
        });

        //添加控件到窗体(的panel)
        panel.add(aboutLabel,BorderLayout.NORTH);
        panel.add(inputArea,BorderLayout.CENTER);
        panel.add(startButton,BorderLayout.SOUTH);

    }
}
