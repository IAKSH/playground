package me.iaksh;

import me.iaksh.mixer.NESLikeSynthesizer;
import me.iaksh.notation.EqualTempNote;
import me.iaksh.notation.Section;
import me.iaksh.player.Player;

import java.util.ArrayList;

public class Main {

    private static ArrayList<Section> genTestSectionSq0() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(3,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(3,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(1,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(2,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,0,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,0,0,1.0f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionSq1() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,2,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(3,2,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(3,2,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,2,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(1,3,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4, 4);
            section.getNotes().add(new EqualTempNote(6, 2, 0, 0.25f, false));
            section.getNotes().add(new EqualTempNote(6, 2, 0, 0.25f, false));
            section.getNotes().add(new EqualTempNote(0, 1, 0, 0.125f, false));
            section.getNotes().add(new EqualTempNote(5, 2, 0, 0.0625f, false));
            section.getNotes().add(new EqualTempNote(6, 2, 0, 0.0625f, false));
            section.getNotes().add(new EqualTempNote(5, 2, 0, 0.125f, false));
            section.getNotes().add(new EqualTempNote(3, 2, 0, 0.125f, false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(2,2,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(2,2,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,2,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,2,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(1,2,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,2,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(2,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(6,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,1.0f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionTri() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,-1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(3,-1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,-1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(1,-2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(2,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(2,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(1,-1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,-1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(2,-1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,-1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(6,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,-1,0,1.0f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionNoi() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(3,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(3,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(1,2,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(3,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(2,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(6,1,0,0.0625f,false));
            section.getNotes().add(new EqualTempNote(5,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,false));
            section.getNotes().add(new EqualTempNote(0,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(1,1,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(2,1,0,0.125f,false));
            section.getNotes().add(new EqualTempNote(7,0,0,0.25f,true));
            section.getNotes().add(new EqualTempNote(6,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new EqualTempNote(6,0,0,1.0f,false));
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