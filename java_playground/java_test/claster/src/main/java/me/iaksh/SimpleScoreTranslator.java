package me.iaksh;

public class SimpleScoreTranslator {

    //private static final float[] frequencies = {0, 261.63f, 293.66f, 329.63f, 349.23f, 392, 440, 493.88f};
    private static final int[] frequencies = {0, 261, 293, 329, 349, 392, 440, 493};

    public static int[] convert(int[] simpleScore) {
        int[] result = new int[simpleScore.length];
        for (int i = 0; i < simpleScore.length; i++) {
            // 将简谱转换为对应的频率
            result[i] = frequencies[simpleScore[i]];
        }
        return result;
    }
}
