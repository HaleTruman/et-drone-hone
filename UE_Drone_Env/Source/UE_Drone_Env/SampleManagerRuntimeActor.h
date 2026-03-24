// Status: in development.
// Native runtime owner for BP_SampleManager orchestration behavior:
// - resolves target drone pawn and hosted sensor component
// - resolves active config reference from ingress owner
// - performs one atomic image+viewpoint capture transaction
// - assembles canonical in-memory observation JSON payload

#pragma once

#include "GameFramework/Actor.h"
#include "SampleManagerRuntimeActor.generated.h"

class APawn;
class UActorComponent;
struct FDroneViewpointSnapshotNative;

UCLASS(BlueprintType, Blueprintable)
class UE_DRONE_ENV_API ASampleManagerRuntimeActor : public AActor
{
    GENERATED_BODY()

public:
    ASampleManagerRuntimeActor();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Sample")
    FString capture_output_mode = TEXT("image_bytes_png");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Sample")
    int32 default_width = 1280;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Sample")
    int32 default_height = 720;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Sample")
    double default_fov_deg = 90.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Discovery")
    FString runtime_discovery_role = TEXT("sample_orchestrator");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Discovery")
    FString runtime_discovery_id = TEXT("BP_SampleManager_Main");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|Discovery")
    FString runtime_discovery_mode = TEXT("deterministic_level_singleton_v1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|ConfigRef")
    FString config_reference_source_role = TEXT("set_config_ingress");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|ConfigRef")
    FString config_reference_source_actor_label = TEXT("BP_SetDataConfig_Main");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|ConfigRef")
    FString config_reference_read_function = TEXT("GetActiveConfigReference");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|ConfigRef")
    FString config_reference_read_mode = TEXT("runtime_query_per_capture_v1");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IO|ConfigRef")
    bool cache_config_reference = false;

    UFUNCTION(BlueprintCallable, Category = "IO|Sample")
    bool ResolveActiveConfigReference(FString& OutConfigId, FString& OutConfigHash, FString& OutError) const;

    UFUNCTION(BlueprintCallable, Category = "IO|Sample")
    bool CaptureNow(
        const FString& RunId,
        const FString& DroneId,
        const FString& CaptureId,
        FString& OutObservationJson,
        FString& OutError
    );

    // Transport-safe wrapper that avoids reflected out-parameter marshalling.
    UFUNCTION(BlueprintCallable, Category = "IO|Sample")
    bool CaptureNowTransport(FString RunId, FString DroneId, FString CaptureId);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "IO|Sample")
    bool CaptureNowTransportEvent(
        const FString& RunId,
        const FString& DroneId,
        const FString& CaptureId
    );
    virtual bool CaptureNowTransportEvent_Implementation(
        const FString& RunId,
        const FString& DroneId,
        const FString& CaptureId
    );

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IO|Sample|Transport")
    bool transport_last_success = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IO|Sample|Transport")
    FString transport_last_observation_json;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IO|Sample|Transport")
    FString transport_last_error;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IO|Sample|Transport")
    FString transport_last_drone_id;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "IO|Sample|Transport")
    FString transport_last_capture_id;

private:
    bool ResolveDronePawn(const FString& DroneId, APawn*& OutPawn, FString& OutError) const;
    bool ReadPrimarySensorComponentName(APawn* Pawn, FString& OutComponentName) const;
    UActorComponent* ResolveSensorComponent(APawn* Pawn) const;
    AActor* ResolveConfigReferenceActor(FString& OutError) const;
    bool InvokeGetActiveConfigReference(
        UObject* Target,
        const FString& FunctionName,
        FString& OutConfigId,
        FString& OutConfigHash,
        FString& OutError
    ) const;
    bool AssembleObservationJson(
        const FString& RunId,
        const FString& DroneId,
        const FString& CaptureId,
        const FString& ConfigId,
        const FString& ConfigHash,
        const FDroneViewpointSnapshotNative& Snapshot,
        const TArray<uint8>& PngBytes,
        int32 CapturedWidth,
        int32 CapturedHeight,
        FString& OutObservationJson,
        FString& OutError
    ) const;
};
