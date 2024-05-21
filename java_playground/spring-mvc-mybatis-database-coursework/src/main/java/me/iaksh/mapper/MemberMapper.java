package me.iaksh.mapper;

import me.iaksh.entity.Member;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface MemberMapper {
    @Select("SELECT * FROM Member WHERE MemberID = #{id}")
    Member getById(@Param("id") long id);

    @Select("SELECT * FROM Member")
    List<Member> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "MemberID", keyColumn = "MemberID")
    @Insert("INSERT INTO Member (Name, MembershipStartDate, MembershipEndDate) " +
            "VALUES (#{member.name}, #{member.membershipStartDate}, #{member.membershipEndDate})")
    void insert(@Param("member") Member member);

    @Update("UPDATE Member SET Name = #{member.name}, MembershipStartDate = #{member.membershipStartDate}, " +
            "MembershipEndDate = #{member.membershipEndDate} WHERE id = #{member.MemberID}")
    void update(@Param("member") Member member);

    @Delete("DELETE FROM Member WHERE MemberID = #{id}")
    void deleteById(@Param("id") long id);
}