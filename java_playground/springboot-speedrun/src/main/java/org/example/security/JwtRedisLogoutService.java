package org.example.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.concurrent.TimeUnit;

@Service
public class JwtRedisLogoutService {

    private final RedisTemplate<String, String> redisTemplate;
    private final JwtService jwtService;

    @Autowired
    public JwtRedisLogoutService(RedisTemplate<String, String> redisTemplate, JwtService jwtService) {
        this.redisTemplate = redisTemplate;
        this.jwtService = jwtService;
    }

    public void addTokenToBlacklist(String token) {
        Date expirationDate = jwtService.getExpirationDateFromToken(token);
        long timeout = expirationDate.getTime() - System.currentTimeMillis();
        redisTemplate.opsForValue().set(token, "blacklisted", timeout, TimeUnit.MILLISECONDS);
    }

    public boolean isTokenBlacklisted(String token) {
        return redisTemplate.hasKey(token);
    }
}
