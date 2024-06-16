package org.example.security;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
public class JwtRedisLogoutService {

    private final RedisTemplate<String, String> redisTemplate;

    public JwtRedisLogoutService(RedisTemplate<String, String> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public void addTokenToBlacklist(String token) {
        redisTemplate.opsForValue().set(token, "blacklisted", 24, TimeUnit.HOURS);
    }

    public boolean isTokenBlacklisted(String token) {
        return redisTemplate.hasKey(token);
    }
}
