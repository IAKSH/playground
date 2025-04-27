package me.iaksh;

import java.sql.*;
import java.util.Scanner;

public class SQLTest {
    private static String user_id = "";
    private static float vote_score = 0.0f;

    public void doIt() throws SQLException {
        Connection connection = DriverManager.getConnection(
                "jdbc:mariadb://192.168.137.153:32768/vote_records",
                "root", "1"
        );

        // get input
        Scanner scan = new Scanner(System.in);
        System.out.println("输入user_id和vote_score：");
        if(scan.hasNext()) {
            user_id = scan.next();
        }
        if(scan.hasNext()) {
            vote_score = scan.nextFloat();
        }
        scan.close();

        // write into DB
        try (PreparedStatement statement = connection.prepareStatement("""
            INSERT INTO records1 (user_id,vote_time,vote_score) VALUE (?,NOW(),?);
        """)) {
            statement.setString(1, user_id);
            statement.setFloat(2, vote_score);
            int rowsInserted = statement.executeUpdate();
        }

        // read from DB
        try (PreparedStatement statement = connection.prepareStatement("""
            SELECT * FROM records1;
        """)) {
            ResultSet resultSet = statement.executeQuery();
            while (resultSet.next()) {
                String name = resultSet.getString("user_id");
                Date time = resultSet.getDate("vote_time");
                float score = resultSet.getFloat("vote_score");
                System.out.printf("用户\"%s\"在[%s]发送了值为[%f]的投票！\r\n",name,time.toString(),score);
            }
        }

        // close DB
        connection.close();
    }
}