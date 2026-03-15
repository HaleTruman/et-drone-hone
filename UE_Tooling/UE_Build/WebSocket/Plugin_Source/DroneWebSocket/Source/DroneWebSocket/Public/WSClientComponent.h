// Desired behavior:
// - Expose a minimal Blueprint-friendly WebSocket client component.
// - Keep transport concerns (connect/send/receive/close) inside C++ and emit clean Blueprint events.
// - Avoid embedding runtime business logic; this component is only the transport pipe.

#pragma once

#include "Components/ActorComponent.h"
#include "WSClientComponent.generated.h"

class IWebSocket;

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FWSClientConnectedSignature);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FWSClientClosedSignature, int32, StatusCode, const FString&, Reason);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FWSClientMessageSignature, const FString&, MessageText);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FWSClientErrorSignature, const FString&, ErrorText);

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class DRONEWEBSOCKET_API UWSClientComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UWSClientComponent();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    FString ServerUrl;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "WebSocket")
    TMap<FString, FString> AdditionalHeaders;

    UPROPERTY(BlueprintAssignable, Category = "WebSocket|Events")
    FWSClientConnectedSignature OnConnected;

    UPROPERTY(BlueprintAssignable, Category = "WebSocket|Events")
    FWSClientClosedSignature OnClosed;

    UPROPERTY(BlueprintAssignable, Category = "WebSocket|Events")
    FWSClientMessageSignature OnMessage;

    UPROPERTY(BlueprintAssignable, Category = "WebSocket|Events")
    FWSClientErrorSignature OnError;

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SetServerUrl(const FString& InServerUrl);

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Connect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    bool IsConnected() const;

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SendTextMessage(const FString& MessageText);

    UFUNCTION(BlueprintCallable, Category = "WebSocket")
    void SendEnvelopeJson(
        const FString& Type,
        const FString& RunId,
        const FString& PayloadJson,
        int32 Seq,
        const FString& DroneId,
        const FString& CaptureId
    );

protected:
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TSharedPtr<IWebSocket> Socket;
    void BindCallbacks();
};
