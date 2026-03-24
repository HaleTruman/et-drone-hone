// Status: in development.
// Runtime ingress/orchestration owner for websocket-driven drone actions.
// This class intentionally lives in the game module so gameplay/runtime ownership
// remains outside the transport plugin boundary.

#pragma once

#include "GameFramework/Actor.h"
#include "SetDataConfigRuntimeActor.generated.h"

class UWSClientComponent;
class FJsonObject;
class AActor;
class APawn;
class UActorComponent;
class ASampleManagerRuntimeActor;

UCLASS(BlueprintType, Blueprintable)
class UE_DRONE_ENV_API ASetDataConfigRuntimeActor : public AActor
{
    GENERATED_BODY()

public:
    ASetDataConfigRuntimeActor();

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

    UFUNCTION(BlueprintCallable, Category = "WebSocket|Config")
    void EmitConfigAck();

    UFUNCTION(BlueprintCallable, Category = "WebSocket|Config")
    void EmitConfigReady();

    UFUNCTION(BlueprintCallable, Category = "WebSocket|Config")
    void EmitConfigError(const FString& ErrorMessage);

    UFUNCTION(BlueprintCallable, Category = "WebSocket|Config")
    bool GetActiveConfigReference(FString& OutConfigId, FString& OutConfigHash) const;

    UFUNCTION(BlueprintNativeEvent, Category = "WebSocket|Config")
    bool OnSetConfigReceived(
        const FString& RunId,
        const FString& ConfigId,
        const FString& ConfigHash,
        const FString& PayloadJson
    );
    virtual bool OnSetConfigReceived_Implementation(
        const FString& RunId,
        const FString& ConfigId,
        const FString& ConfigHash,
        const FString& PayloadJson
    );

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
    friend class FTransportDispatchStep11Test;

    struct FSensorRigRuntimeConfig
    {
        FName ActiveViewpoint = FName(TEXT("front"));
        double FrontFovDeg = 90.0;
        int32 CaptureWidth = 1280;
        int32 CaptureHeight = 720;
        FVector FrontOffsetCm = FVector::ZeroVector;
        FRotator FrontRotationDeg = FRotator::ZeroRotator;
    };

    int32 OutSeq = 1;
    bool bConfigResponseSentForCurrentSetConfig = false;
    bool bHasActiveSensorRigConfig = false;
    FSensorRigRuntimeConfig ActiveSensorRigConfig;
    TMap<FString, TWeakObjectPtr<APawn>> SpawnedDrones;

    void ClearActiveConfigReference();
    void InvokeBlueprintHookIfExists(const TCHAR* FunctionName);
    bool ExtractSensorRigConfig(
        const TSharedPtr<FJsonObject>& PayloadObject,
        FSensorRigRuntimeConfig& OutConfig,
        FString& OutError
    ) const;
    bool ReadPrimarySensorComponentName(APawn* Pawn, FString& OutComponentName) const;
    UActorComponent* ResolveSensorComponent(APawn* Pawn) const;
    bool ApplySensorRigConfigToTarget(UObject* Target, FString& OutError) const;
    bool ApplySensorRigConfigToPawn(APawn* Pawn, FString& OutError) const;
    void ApplySensorRigConfigToPawnIfAvailable(const FString& RunId, APawn* Pawn);
    bool SetNameLikeProperty(UObject* Target, const TCHAR* PropertyName, const FName& Value) const;
    bool SetRealLikeProperty(UObject* Target, const TCHAR* PropertyName, double Value) const;
    bool SetIntLikeProperty(UObject* Target, const TCHAR* PropertyName, int32 Value) const;
    bool SetBoolLikeProperty(UObject* Target, const TCHAR* PropertyName, bool Value) const;
    bool GetBoolLikeProperty(const UObject* Target, const TCHAR* PropertyName, bool& OutValue) const;
    bool GetStringLikeProperty(const UObject* Target, const TCHAR* PropertyName, FString& OutValue) const;

    void SendEnvelope(
        const FString& Type,
        const FString& RunId,
        const TSharedPtr<FJsonObject>& PayloadObject,
        const FString& DroneId = TEXT(""),
        const FString& CaptureId = TEXT("")
    );
    void SendError(const FString& RunId, const FString& ErrorMessage);
    bool ParseRootObject(const FString& MessageText, TSharedPtr<FJsonObject>& OutRoot, FString& OutError) const;
    bool TryGetStringField(const TSharedPtr<FJsonObject>& Root, const TCHAR* Key, FString& OutValue) const;
    ASampleManagerRuntimeActor* ResolveSampleManagerActor(FString& OutError) const;
    bool InvokeSampleManagerCaptureNow(
        ASampleManagerRuntimeActor* Target,
        const FString& RunId,
        const FString& DroneId,
        const FString& CaptureId,
        TSharedPtr<FJsonObject>& OutObservationPayload,
        FString& OutResolvedDroneId,
        FString& OutResolvedCaptureId,
        FString& OutError
    ) const;
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
