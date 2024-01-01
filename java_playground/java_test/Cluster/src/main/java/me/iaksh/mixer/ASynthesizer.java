package me.iaksh.mixer;

import me.iaksh.oscillator.*;
import me.iaksh.notation.Section;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class ASynthesizer extends Synthesizer implements Exporter {

    public ASynthesizer(int bpm) {
        for(int i = 0;i < 4;i++)
            tracks.add(new Track(bpm));
    }

    @Override
    public void saveToWav(String path,ArrayList<ArrayList<Section>> sections) {
        short[] waveform = genWavform(sections);
        byte[] byteBuffer = new byte[waveform.length * 2];
        ByteBuffer.wrap(byteBuffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(waveform);

        AudioFormat format = new AudioFormat(Oscillator.getSampleRate(), 16, 1, true, false);

        ByteArrayInputStream bais = new ByteArrayInputStream(byteBuffer);
        AudioInputStream audioInputStream = new AudioInputStream(bais, format, waveform.length);

        File file = new File(path);
        try {
            AudioSystem.write(audioInputStream, AudioFileFormat.Type.WAVE, file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public short[] genWavform(ArrayList<ArrayList<Section>> sections) {
        ArrayList<ArrayList<Short>> channels = new ArrayList<>();

        channels.add(tracks.get(0).genWaveform(new ExpGradientEffect(new SquareOscillator(),4.0f),sections.get(0)));
        channels.add(tracks.get(1).genWaveform(new ExpGradientEffect(new SquareOscillator(),4.0f),sections.get(1)));
        channels.add(tracks.get(2).genWaveform(new ExpGradientEffect(new TriangleOscillator(),4.0f),sections.get(2)));
        channels.add(tracks.get(3).genWaveform(new ExpGradientEffect(new NoiseOscillator(),4.0f),sections.get(3)));

        return Mixer.mix(channels);
    }
}
