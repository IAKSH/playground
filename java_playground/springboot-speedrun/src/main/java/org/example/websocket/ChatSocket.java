package org.example.websocket;

import jakarta.websocket.*;
import jakarta.websocket.server.ServerEndpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Instant;

@ServerEndpoint("/chat/websocket")
public class ChatSocket {

    private static final Logger LOGGER = LoggerFactory.getLogger(ChatSocket.class);

    private Session session;

    @OnMessage
    public void onMessage(String message) throws IOException {
        LOGGER.info("[websocket] received message: id={}, message={}",this.session.getId(),message);

        if(message.equalsIgnoreCase("bye")) {
            // 由服务器主动关闭连接。状态码为 NORMAL_CLOSURE（正常关闭）。
            this.session.close(new CloseReason(CloseReason.CloseCodes.NORMAL_CLOSURE, "Bye"));;
            return;
        }

        this.session.getAsyncRemote().sendText("["+ Instant.now().toEpochMilli() +"] server received: " + message);
    }

    @OnOpen
    public void onOpen(Session session, EndpointConfig endpointConfig){
        // 保存 session 到对象
        this.session = session;
        LOGGER.info("[websocket] new connection：id={}", this.session.getId());
    }

    @OnClose
    public void onClose(CloseReason closeReason){
        LOGGER.info("[websocket] disconnected：id={}，reason={}", this.session.getId(),closeReason);
    }

    @OnError
    public void onError(Throwable throwable) throws IOException {

        LOGGER.info("[websocket] connect error：id={}，throwable={}", this.session.getId(), throwable.getMessage());

        // 关闭连接。状态码为 UNEXPECTED_CONDITION（意料之外的异常）
        this.session.close(new CloseReason(CloseReason.CloseCodes.UNEXPECTED_CONDITION, throwable.getMessage()));
    }
}
