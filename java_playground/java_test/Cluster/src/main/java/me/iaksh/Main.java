package me.iaksh;

import me.iaksh.cluster.OpenALLoader;
import me.iaksh.mixer.NESLikePlayer;
import me.iaksh.mixer.Player;
import me.iaksh.notation.Note;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public class Main {

    private static final int[] a = {6,2,3,5,3,2};

    private static ArrayList<Section> genTestSectionSq0() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(5,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(5,-1,-1,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(5,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-1,0,0.25f,false));
            section.getNotes().add(new Note(1,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionSq1() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(6,1,0,0.125f,false));
            section.getNotes().add(new Note(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.25f,false));
            section.getNotes().add(new Note(3,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(5,0,-1,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(6,1,0,0.125f,false));
            section.getNotes().add(new Note(5,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.25f,false));
            section.getNotes().add(new Note(3,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(7,0,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            section.getNotes().add(new Note(2,1,0,0.125f,false));
            section.getNotes().add(new Note(1,1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,0,0,0.25f,false));
            section.getNotes().add(new Note(1,1,0,0.25f,false));
            section.getNotes().add(new Note(2,1,0,0.25f,false));
            section.getNotes().add(new Note(3,1,0,0.25f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionTri() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(6,-2,0,0.25f,false));
            section.getNotes().add(new Note(3,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(5,-2,-1,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(5,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(6,-2,0,0.25f,false));
            section.getNotes().add(new Note(3,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-2,0,0.125f,false));
            section.getNotes().add(new Note(7,-2,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            section.getNotes().add(new Note(2,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-2,0,0.25f,false));
            section.getNotes().add(new Note(1,-1,0,0.25f,false));
            section.getNotes().add(new Note(2,-1,0,0.25f,false));
            section.getNotes().add(new Note(3,-1,0,0.25f,false));
            sections.add(section);
        }

        return sections;
    }

    private static ArrayList<Section> genTestSectionNoi() {
        ArrayList<Section> sections = new ArrayList<>();

        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(5,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(5,-1,-1,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,0,0,0.125f,false));
            section.getNotes().add(new Note(5,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(6,-1,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(6,-1,0,0.125f,false));
            section.getNotes().add(new Note(7,-1,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.125f,false));
            section.getNotes().add(new Note(1,0,0,0.125f,false));
            sections.add(section);
        }
        {
            Section section = new Section(4,4);
            section.getNotes().add(new Note(7,-1,0,0.25f,false));
            section.getNotes().add(new Note(1,0,0,0.25f,false));
            section.getNotes().add(new Note(2,0,0,0.25f,false));
            section.getNotes().add(new Note(3,0,0,0.25f,false));
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
        OpenALLoader loader = new OpenALLoader();
        Player player = new NESLikePlayer(140);
        player.play(genTestSection());
    }
}