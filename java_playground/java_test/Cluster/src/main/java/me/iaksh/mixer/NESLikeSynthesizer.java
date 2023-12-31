package me.iaksh.mixer;

import me.iaksh.oscillator.Oscillator;
import me.iaksh.oscillator.SquareOscillator;
import me.iaksh.oscillator.TriangleOscillator;
import me.iaksh.oscillator.NoiseOscillator;
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

public class NESLikeSynthesizer extends Synthesizer implements MusicPlayer,Exporter{

    public NESLikeSynthesizer(int bpm) {
        for(int i = 0;i < 4;i++)
            tracks.add(new Track(bpm));
    }

    @Override
    public void play(ArrayList<ArrayList<Section>> sections) {
        if(sections.size() != 4)
            throw new IllegalArgumentException();

        tracks.get(0).genWaveform(new SquareOscillator(),sections.get(0));
        tracks.get(1).genWaveform(new SquareOscillator(),sections.get(1));
        tracks.get(2).genWaveform(new TriangleOscillator(),sections.get(2));
        tracks.get(3).genWaveform(new NoiseOscillator(),sections.get(3));
        System.out.println("genWaveform() finished!");
        for(Track track : tracks)
            track.play();

        waitAllTrackToBeFinished();
        System.out.println("All track finished!");
        destroyTracks();
    }

    @Override
    public void play(float gain,ArrayList<ArrayList<Section>> sections) {
        for(Track track : tracks)
            track.setGain(gain);
        play(sections);
    }

    @Override
    public void saveToWav(String path,ArrayList<ArrayList<Section>> sections) {
        ArrayList<ArrayList<Short>> channels = new ArrayList<>();
        channels.add(tracks.get(0).genWaveform(new SquareOscillator(),sections.get(0)));
        channels.add(tracks.get(1).genWaveform(new SquareOscillator(),sections.get(1)));
        channels.add(tracks.get(2).genWaveform(new TriangleOscillator(),sections.get(2)));
        channels.add(tracks.get(3).genWaveform(new NoiseOscillator(),sections.get(3)));
        short[] waveform = Mixer.mix(channels);

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
}
