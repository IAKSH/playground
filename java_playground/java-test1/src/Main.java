import java.util.ArrayList;

public class Main {
    private static String getInputStr(ConsoleInputer inputer) {
        char inputChar = 0;
        StringBuilder inputStr = new StringBuilder();
        while(inputChar != '=') {
            inputChar = inputer.readChar();
            if(inputChar != '\n') {
                inputStr.append(inputChar);
            }
        }
        return inputStr.toString();
    }

    private static boolean isComputOptionChar(char c) {
        return c == '+' || c == '-' || c == '*' || c == '\\';
    }

    private static boolean isComputValueChar(char c) {
        return c >= '0' && c <= '9';
    }

    private static int getBaseNumberFromStr(String str) {
        StringBuilder intStr = new StringBuilder();
        for(int i = 0;i < str.length();i++) {
            char c = str.charAt(i);
            if(c != ' ') {
                if(!isComputValueChar(c)) {
                    return Integer.parseInt(intStr.toString());
                }
                intStr.append(c);
            }
        }
        throw new RuntimeException("base number not found");
    }

    private static void parseComputOptionsFromStr(ArrayList<ComputAction> actions, String str) {
        for(int i = 0;i < str.length();i++) {
            char c = str.charAt(i);
            if(c != ' ' && isComputOptionChar(c)) {
                char actionChar = c;
                StringBuilder intStr = new StringBuilder();
                for(int j = i + 1;j < str.length();j++) {
                    c = str.charAt(j);
                    if(!isComputValueChar(c)) {
                        int actionValue = Integer.parseInt(intStr.toString());
                        actions.add(new ComputAction(actionValue, actionChar));
                        i = j;
                        break;
                    }
                    intStr.append(c);
                }
            }
        }
    }

    private static int computAllActions(ArrayList<ComputAction> actions,int baseNum) {
        int result = baseNum;
        for(var action : actions) {
            result = action.executeWithBaseNum(result);
        }
        actions.clear();
        return result;
    }

    public static void main(String[] args) {
        System.out.println("草泥马");
        /*
        ConsoleInputer inputer = new ConsoleInputer();
        ArrayList<ComputAction> actions = new ArrayList<ComputAction>();
        while(true) {
            String str = getInputStr(inputer);
            int baseNum = getBaseNumberFromStr(str);
            parseComputOptionsFromStr(actions,str);
            System.out.println(computAllActions(actions,baseNum));
        }
         */
    }
}
