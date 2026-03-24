// Desired behavior:
// - Provide transport-only websocket actor compatibility in the plugin.
// - Do not own gameplay/runtime orchestration (config apply, spawn, cmd, capture).
// - Expose connection/message hooks for game-module runtime owners when needed.

#pragma once

#include "GameFramework/Actor.h"
#include "WSConfigHandshakeActor.generated.h"

class UWSClientComponent;
class FJsonObject;

UCLASS(BlueprintType, Blueprintable)
class DRONEWEBSOCKET_API AWSConfigHandshakeActor : public AActor
{
    GENERATED_BODY()

public:
    AWSConfigHandshakeActor();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    FString ServerUrl;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    bool bAutoConnectOnBeginPlay = true;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    FString LastSocketError;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "WebSocket")
    TObjectPtr<UWSClientComponent> WSClient;

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Connect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SendEnvelope(
        const FString& Type,
        const FString& RunId,
        const FString& PayloadJson,
        int32 Seq,
        const FString& DroneId = TEXT(""),
        const FString& CaptureId = TEXT("")
    );

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SendErrorEnvelope(const FString& RunId, const FString& ErrorMessage, int32 Seq);

    UFUNCTION(BlueprintNativeEvent, Category = "WebSocket|Events")
    void OnTransportMessage(const FString& MessageText);
    virtual void OnTransportMessage_Implementation(const FString& MessageText);

protected:
    virtual void BeginPlay() override;

    UFUNCTION()
    void HandleSocketConnected();

    UFUNCTION()
    void HandleSocketClosed(int32 StatusCode, const FString& Reason);

    UFUNCTION()
    void HandleSocketError(const FString& ErrorText);

    UFUNCTION()
    void HandleSocketMessage(const FString& MessageText);
};
