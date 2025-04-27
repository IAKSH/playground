#pragma once

namespace cppadv::input {
    
    enum class Key {
        A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,
        a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
        Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8,Num9,Num0,
        F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,Del,
        Tab,Caps,LShift,LCtrl,LAlt,RAlt,RCtrl,RShift,Enter,Backspace,Space
    };

    void initialize();

    bool isKeyUp(Key k);
    bool isKeyDown(Key k);

    int getMousePositionX();
    int getMousePositionY();
    bool isMouseLeftClick();
    bool isMouseRightClick();
    bool isMouseMiddleClick();
}