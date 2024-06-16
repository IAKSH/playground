package org.example.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.stereotype.Service;

import java.security.Key;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
public class JwtService {
    // Token有效期限
    @Value("${conf.token.expiration}") // 透過文件配置的方式給值，詳細教學在此系列的第7天文章
    private Long EXPIRATION_TIME; //單位ms
    // 上述兩行可以改寫為下面這行
    // private Long EXPIRATION_TIME = 900000L

    @Value("${conf.token.secret}") // 透過文件配置的方式給值
    private String SECRET_KEY;
    // 上述兩行可以改寫為下面這行
    // private String SECRET_KEY = "你的私鑰" //在這個範例中我使用的簽名算法為(HS256)"SignatureAlgorithm.HS256"，我們可以透過線上的密碼產生器，產生長度64的任意字元組成的字串。注意!如果你使用的是其他算法，則你需要給定該算法對應的私鑰規則，具體可以上網查詢

    public String extractUsername(String token) {
        try {
            return extractClaim(token, Claims::getSubject);
        }catch (ExpiredJwtException e){
            return e.getClaims().getSubject();
        }
    }

    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }

    public String generateToken(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("sub", userDetails.getUsername());
        claims.put("roles", userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList()));
        return generateToken(claims, userDetails);
    }

    private String generateToken(Map<String, Object> claims, UserDetails userDetails) {
        return Jwts.builder()
                .setClaims(claims)
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + EXPIRATION_TIME))
                .signWith(getSignInKey(), SignatureAlgorithm.HS256)
                .compact();
    }

    /**
     * 驗證Token有效性，比對JWT和UserDetails的Username(Email)是否相同
     * @return 有效為True，反之False
     */
    public boolean isTokenValid(String token, UserDetails userDetails) {
        final String username = extractUsername(token);
        return (username.equals(userDetails.getUsername())) && !isTokenExpired(token);
    }

    private boolean isTokenExpired(String token) {
        final Date expirationDate = extractExpiration(token);
//        return extractExpiration(token).before(new Date());
        return expirationDate != null && expirationDate.before(new Date());
    }

    private Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }

    /**
     * 獲取令牌中所有的聲明
     * @return 令牌中所有的聲明
     */
    public Claims extractAllClaims(String token) {
        try {
            return Jwts
                    .parser()
                    .setSigningKey(getSignInKey())
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
        }
        catch (ExpiredJwtException e){
            return e.getClaims();
        }
    }

    private Key getSignInKey() {
        byte[] keyBytes = Decoders.BASE64.decode(SECRET_KEY);
        return Keys.hmacShaKeyFor(keyBytes);
    }
}