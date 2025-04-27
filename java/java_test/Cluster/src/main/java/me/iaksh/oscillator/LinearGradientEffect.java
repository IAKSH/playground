package me.iaksh.oscillator;

public class LinearGradientEffect extends GradientEffect {
    public LinearGradientEffect(Oscillator oscillator) {
        super(oscillator);
    }

    @Override
    protected float gradientCoefficient(int waveformLen,int i) {
        return 1.0f - (float) i / (float) waveformLen;
    }
}
