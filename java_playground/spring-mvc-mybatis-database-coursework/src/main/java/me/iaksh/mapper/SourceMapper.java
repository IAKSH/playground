package me.iaksh.mapper;

import me.iaksh.entity.Source;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface SourceMapper {
    @Select("SELECT * FROM Source WHERE SourceID = #{id}")
    Source getById(@Param("id") long id);

    @Select("SELECT * FROM Source")
    List<Source> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "SourceID", keyColumn = "SourceID")
    @Insert("INSERT INTO Source (Name) " +
            "VALUES (#{source.name})")
    void insert(@Param("source") Source source);

    @Update("UPDATE Source SET Name = #{source.name} WHERE id = #{source.SourceID}")
    void update(@Param("source") Source source);

    @Delete("DELETE FROM Source WHERE SourceID = #{id}")
    void deleteById(@Param("id") long id);
}
