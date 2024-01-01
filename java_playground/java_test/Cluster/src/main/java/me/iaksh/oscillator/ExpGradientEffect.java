package me.iaksh.oscillator;

public class ExpGradientEffect extends GradientEffect {

    private float expCoefficient = 1;

    public ExpGradientEffect(Oscillator oscillator) {
        super(oscillator);
    }

    @Override
    protected float gradientCoefficient(int waveformLen, int i) {
        // y=e^{-ax}, a = expCoefficient
        return (float) Math.exp(-expCoefficient * ((float) i / (float) waveformLen));
    }

    public ExpGradientEffect(Oscillator oscillator, float expCoefficient) {
        super(oscillator);
        this.expCoefficient = expCoefficient;
    }

    public float getExpCoefficient() {
        return expCoefficient;
    }

    public void setExpCoefficient(float expCoefficient) {
        this.expCoefficient = expCoefficient;
    }
}
