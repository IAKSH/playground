package me.iaksh.oscillator;

public abstract class GradientEffect extends Effect {

    public GradientEffect(Oscillator oscillator) {
        super(oscillator);
    }

    protected abstract float gradientCoefficient(int waveformLen,int i);

    @Override
    public final short[] genWaveform(int ms, int freq) {
        short[] waveform = oscillator.genWaveform(ms,freq);
        for(int i = 0;i < waveform.length;i++) {
            waveform[i] = (short) (waveform[i] * gradientCoefficient(waveform.length,i));
        }
        return waveform;
    }
}
