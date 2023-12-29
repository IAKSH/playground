package me.iaksh;

public class Main {
    private Mixer mixer;
    private Channel[] channels;
    private SquareWave squareWave0;
    private SquareWave squareWave1;
    private TriangleWave triangleWave;
    //private NoiseWave noiseWave;

    void initMixer() {
        mixer = new Mixer();
    }

    void initWaves() {
        squareWave0 = new SquareWave(300);
        squareWave1 = new SquareWave(300);
        triangleWave = new TriangleWave(300);
        //noiseWave = new NoiseWave(100);
    }

    void initChannels() {
        channels = new Channel[4];

        channels[0] = new Channel() {
            @Override
            public void run() {
                claster.setGain(0.1f);
                try {
                    while(true) {
                        int[] squareWaveFreqs = {523,587,659,523,659,784};
                        for(int i = 0;i < squareWaveFreqs.length;i++) {
                            squareWave0.setFreq(squareWaveFreqs[i]);
                            squareWave0.generate(44100);
                            claster.bindWave(squareWave0);
                            claster.start();
                            Thread.sleep(squareWave0.getDurationMs());
                        }
                    }
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        channels[1] = new Channel() {
            @Override
            public void run() {
                claster.setGain(0.2f);
                try {
                    while(true) {
                        int[] squareWaveFreqs = {1,659,1,659,1,523};
                        for(int i = 0;i < squareWaveFreqs.length;i++) {
                            squareWave1.setFreq(squareWaveFreqs[i]);
                            squareWave1.generate(44100);
                            claster.bindWave(squareWave1);
                            claster.start();
                            Thread.sleep(squareWave1.getDurationMs());
                        }
                    }
                }
                catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        channels[2] = new Channel() {
            @Override
            public void run() {
                claster.setGain(0.3f);
                try {
                    while(true) {
                        int[] triangleWaveFreqs = {523,587,659,523,659,784};
                        for(int i = 0;i < triangleWaveFreqs.length;i++) {
                            triangleWave.setFreq(triangleWaveFreqs[i]);
                            triangleWave.generate(44100);
                            claster.bindWave(triangleWave);
                            claster.start();
                            Thread.sleep(triangleWave.getDurationMs());
                        }
                    }
                }
                catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        channels[3] = new Channel() {
            @Override
            public void run() {
                /*
                claster.setGain(0.1f);
                try {
                    while(true) {
                        int[] noiseWaveFreq = {1,659,1,659,1,523};
                        for(int i = 0;i < noiseWaveFreq.length;i++) {
                            noiseWave.setFreq(noiseWaveFreq[i]);
                            noiseWave.generate(44100);
                            claster.bindWave(noiseWave);
                            claster.start();
                            Thread.sleep(noiseWave.getDurationMs());
                            claster.stop();
                        }
                    }
                }
                catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                 */
            }
        };
    }

    void startAllChannel() {
        for(Channel channel : channels)
            new Thread(channel).start();
    }

    void destroy() {
        for(Channel channel : channels)
            channel.destoryClaster();
        mixer.destoryOpenAL();
    }

    void start() throws InterruptedException {
        initMixer();
        initWaves();
        initChannels();
        startAllChannel();
        while(true)
            Thread.sleep(1);
    }

    public static void main(String[] args) {
        try {
            Main m = new Main();
            m.start();
            m.destroy();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}