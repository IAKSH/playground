package me.iaksh;

import me.iaksh.mixer.NESLikeSynthesizer;
import me.iaksh.notation.EqualTempNote;
import me.iaksh.notation.FreqNote;
import me.iaksh.notation.Section;
import me.iaksh.player.Player;

import java.util.ArrayList;

public class Main {

    private static ArrayList<Section> genTestSectionSq0() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,3,1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,1,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,2,1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,2,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,1,1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,1,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,0,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,1,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,1,0,0));
            section.getNotes().add(new EqualTempNote(0.25f,true,7,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,0,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1.0f,false,6,0,0));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionSq1() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,2,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,2,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,3,2,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,2,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,2,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,1,3,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            sections.add(section);
        }
        {
            Section section = new Section(4, 4);
            section.getNotes().add(new EqualTempNote(0.25f, false,6, 2, 0));
            section.getNotes().add(new EqualTempNote(0.25f, false,6, 2, 0));
            section.getNotes().add(new EqualTempNote(0.125f, false,0, 1, 0));
            section.getNotes().add(new EqualTempNote(0.0625f, false,5, 2, 0));
            section.getNotes().add(new EqualTempNote(0.0625f, false,6, 2, 0));
            section.getNotes().add(new EqualTempNote(0.125f, false,5, 2, 0));
            section.getNotes().add(new EqualTempNote(0.125f, false,3, 2, 0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,2,2,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,2,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,2,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,2,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,1,2,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,1,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,1,2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,2,0));
            section.getNotes().add(new EqualTempNote(0.25f,true,7,1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1.0f,false,6,1,0));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionTri() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,1,-2,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,0,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,3,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,2,-1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,2,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.0625f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,5,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,false,1,-1,0));
            section.getNotes().add(new EqualTempNote(0.25f,false,1,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,0,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,7,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0.25f,true,1,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,2,-1,0));
            section.getNotes().add(new EqualTempNote(0.25f,true,7,-1,0));
            section.getNotes().add(new EqualTempNote(0.125f,false,6,-1,0));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1.0f,false,6,-1,0));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionNoi() {
        ArrayList<Section> sections = new ArrayList<>();

        for(int i = 0;i < 9;i++) {
            Section section = new Section(4,4);
            section.getNotes().add(new FreqNote(0.21875f,false,0));
            section.getNotes().add(new FreqNote(0.03125f,false,1000));
            section.getNotes().add(new FreqNote(0.21875f,false,0));
            section.getNotes().add(new FreqNote(0.03125f,false,100));
            section.getNotes().add(new FreqNote(0.21875f,false,0));
            section.getNotes().add(new FreqNote(0.03125f,false,10));
            section.getNotes().add(new FreqNote(0.21875f,false,0));
            section.getNotes().add(new FreqNote(0.03125f,false,1));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<ArrayList<Section>> genTestSection() {
        ArrayList<ArrayList<Section>> sections = new ArrayList<>();
        sections.add(genTestSectionSq0());
        sections.add(genTestSectionSq1());
        sections.add(genTestSectionTri());
        sections.add(genTestSectionNoi());
        return sections;
    }

    public static void main(String[] args) {
        //new NESLikeSynthesizer(120).saveToWav("./out.wav",genTestSection());
        new Player().play(0.05f,new NESLikeSynthesizer(120).genWavform(genTestSection()));
        Player.closeOpenAL();
    }
}