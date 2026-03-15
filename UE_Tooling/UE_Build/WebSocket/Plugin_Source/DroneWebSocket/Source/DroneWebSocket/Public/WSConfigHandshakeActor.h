// Desired behavior:
// - Provide a minimal UE runtime actor that performs WebSocket `SET_CONFIG` handshake.
// - Keep handshake semantics focused on config receipt/ready signaling (ACK -> CONFIG_READY).
// - Expose deterministic runtime state so Blueprint wrappers can gate later systems on config readiness.

#pragma once

#include "GameFramework/Actor.h"
#include "WSConfigHandshakeActor.generated.h"

class UWSClientComponent;
class FJsonObject;
class APawn;

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

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    bool bFailOnRunIdMismatch = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    FString ExpectedRunId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Runtime|Drone")
    FString DronePawnClassPath = TEXT("/Game/Drone_Content/Blueprints/BP_DronePawn.BP_DronePawn_C");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Runtime|Drone")
    float SpawnSpacingCm = 250.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Runtime|Drone")
    float CommandTranslationScaleCm = 25.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Runtime|Drone")
    float CommandRotationScaleDeg = 5.0f;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    bool bConfigReceived = false;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    bool bConfigReady = false;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    FString ActiveRunId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    FString ActiveConfigId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    FString ActiveConfigHash;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|State")
    FString LastError;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "WebSocket")
    TObjectPtr<UWSClientComponent> WSClient;

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Connect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Disconnect();

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

private:
    int32 OutSeq = 1;
    TMap<FString, TWeakObjectPtr<APawn>> SpawnedDrones;

    void SendEnvelope(
        const FString& Type,
        const FString& RunId,
        const TSharedPtr<FJsonObject>& PayloadObject,
        const FString& DroneId = TEXT(""),
        const FString& CaptureId = TEXT("")
    );
    void SendAckAndConfigReady(const FString& RunId, const FString& ConfigId, const FString& ConfigHash);
    void SendError(const FString& RunId, const FString& ErrorMessage);
    bool ParseRootObject(const FString& MessageText, TSharedPtr<FJsonObject>& OutRoot, FString& OutError) const;
    bool TryGetStringField(const TSharedPtr<FJsonObject>& Root, const TCHAR* Key, FString& OutValue) const;
    void HandleActionMessage(
        const FString& MessageType,
        const FString& RunId,
        const TSharedPtr<FJsonObject>& Root,
        const TSharedPtr<FJsonObject>& PayloadObject
    );
    void HandleSpawnDrones(const FString& RunId, const TSharedPtr<FJsonObject>& PayloadObject);
    void HandleCmd(const FString& RunId, const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& PayloadObject);
    void HandleCaptureNow(const FString& RunId, const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& PayloadObject);
    APawn* ResolveDroneById(const FString& DroneId) const;
    FString NextDroneId() const;
};
