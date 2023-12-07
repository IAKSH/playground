package me.iaksh.hexagram;

public interface Casting {
    int[] yao_values = new int[6];

    void exec();

    default Hexagram getHostHexagram() {
        int code = 0b000000;
        for(var value : yao_values) {
            code = code << 1 | ((value % 2 != 0) ? 1 : 0);
        }
        return Hexagram.fromCode(code);
    }

    default Hexagram getTransHexagram() {
        int code = 0b000000;
        for(var value : yao_values) {
            if(value == 6) {
                value = 7;
            }
            else if(value == 9) {
                value = 8;
            }
            code = code << 1 | ((value % 2 != 0) ? 1 : 0);
        }
        return Hexagram.fromCode(code);
    }
}
