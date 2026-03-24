// Desired behavior:
// - Define lightweight protocol structs/helpers for Blueprint runtime usage.
// - Keep message handling transport-focused (string JSON in/out) without embedding business logic.
// - Support both outgoing envelope generation and incoming envelope parsing in UE runtime.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "WSProtocolTypes.generated.h"

USTRUCT(BlueprintType)
struct DRONEWEBSOCKET_API FWSBridgeEnvelope
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString Type;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString RunId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString SchemaVersion;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    int32 Seq = 0;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString Timestamp;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString CaptureId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString MessageId;

    UPROPERTY(BlueprintReadOnly, Category = "WebSocket|Protocol")
    FString PayloadJson;
};

UCLASS()
class DRONEWEBSOCKET_API UWSProtocolTypes : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "WebSocket|Protocol")
    static FString BuildEnvelopeJson(
        const FString& Type,
        const FString& RunId,
        const FString& PayloadJson,
        int32 Seq,
        const FString& DroneId,
        const FString& CaptureId
    );

    UFUNCTION(BlueprintCallable, Category = "WebSocket|Protocol")
    static bool ParseEnvelopeJson(
        const FString& MessageJson,
        FWSBridgeEnvelope& OutEnvelope,
        FString& OutError
    );
};
