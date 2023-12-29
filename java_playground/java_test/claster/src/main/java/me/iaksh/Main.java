package me.iaksh;

import java.nio.*;

public class Main {
    public static void main(String[] args) {
        try {
            Mixer mixer = new Mixer();
            Claster[] clasters = new Claster[3];
            for(int i = 0;i < clasters.length;i++)
                clasters[i] = new Claster();

            SquareWave[] squareWaves = new SquareWave[6];
            int[] squareWaveFreqs = {523,587,659,523,659,784};
            for(int i = 0;i < squareWaves.length;i++) {
                SquareWave wave = new SquareWave(300);
                wave.setFreq(squareWaveFreqs[i]);
                wave.generate(44100);
                squareWaves[i] = wave;
            }

            TriangleWave[] triangleWaves = new TriangleWave[3];
            int[] triangleWaveFreqs = {1047,659,523};
            for(int i = 0;i < triangleWaves.length;i++) {
                TriangleWave wave = new TriangleWave(300);
                wave.setFreq(triangleWaveFreqs[i]);
                wave.generate(44100);
                triangleWaves[i] = wave;
            }

            NoiseWave noiseWave = new NoiseWave(300);
            noiseWave.setFreq(300);
            noiseWave.generate(44100);
            clasters[2].bindWave(noiseWave);

            for(int i = 0;i < 6;i++) {
                clasters[2].start();
                clasters[0].bindWave(squareWaves[i]);
                Thread.sleep(10 * i);
                clasters[2].stop();
                clasters[0].start();
                clasters[1].bindWave(triangleWaves[i % 3]);
                clasters[1].start();
                Thread.sleep(500);
            }
            clasters[0].stop();
            Thread.sleep(1000);
            clasters[1].stop();

            for(Claster claster : clasters)
                claster.destroyAlSource();
            mixer.destoryOpenAL();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}