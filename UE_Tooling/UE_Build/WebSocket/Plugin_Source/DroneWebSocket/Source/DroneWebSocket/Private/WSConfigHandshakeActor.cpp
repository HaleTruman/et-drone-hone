// Desired behavior:
// - Keep plugin actor transport-only.
// - Do not own gameplay/runtime orchestration.

#include "WSConfigHandshakeActor.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "WSClientComponent.h"

AWSConfigHandshakeActor::AWSConfigHandshakeActor()
{
    PrimaryActorTick.bCanEverTick = false;
    ServerUrl = TEXT("ws://127.0.0.1:8765");
    WSClient = CreateDefaultSubobject<UWSClientComponent>(TEXT("WSClient"));
}

void AWSConfigHandshakeActor::BeginPlay()
{
    Super::BeginPlay();

    if (!WSClient)
    {
        LastSocketError = TEXT("WSClient component missing.");
        UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] %s"), *LastSocketError);
        return;
    }

    WSClient->OnConnected.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketConnected);
    WSClient->OnClosed.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketClosed);
    WSClient->OnError.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketError);
    WSClient->OnMessage.AddDynamic(this, &AWSConfigHandshakeActor::HandleSocketMessage);

    if (bAutoConnectOnBeginPlay)
    {
        Connect();
    }
}

void AWSConfigHandshakeActor::Connect()
{
    if (!WSClient)
    {
        return;
    }
    WSClient->SetServerUrl(ServerUrl);
    WSClient->Connect();
}

void AWSConfigHandshakeActor::Disconnect()
{
    if (WSClient)
    {
        WSClient->Disconnect();
    }
}

void AWSConfigHandshakeActor::SendEnvelope(
    const FString& Type,
    const FString& RunId,
    const FString& PayloadJson,
    int32 Seq,
    const FString& DroneId,
    const FString& CaptureId
)
{
    if (!WSClient || !WSClient->IsConnected())
    {
        UE_LOG(LogTemp, Warning, TEXT("[WSConfigHandshakeActor] cannot send %s: websocket not connected"), *Type);
        return;
    }

    WSClient->SendEnvelopeJson(Type, RunId, PayloadJson, Seq, DroneId, CaptureId);
}

void AWSConfigHandshakeActor::SendErrorEnvelope(const FString& RunId, const FString& ErrorMessage, int32 Seq)
{
    TSharedPtr<FJsonObject> ErrorPayload = MakeShared<FJsonObject>();
    ErrorPayload->SetStringField(TEXT("message"), ErrorMessage);
    ErrorPayload->SetStringField(TEXT("source"), TEXT("DroneWebSocketTransport"));

    FString PayloadJson = TEXT("{}");
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadJson);
    FJsonSerializer::Serialize(ErrorPayload.ToSharedRef(), Writer);
    SendEnvelope(TEXT("ERROR"), RunId, PayloadJson, Seq);
}

void AWSConfigHandshakeActor::OnTransportMessage_Implementation(const FString& MessageText)
{
    UE_LOG(LogTemp, Verbose, TEXT("[WSConfigHandshakeActor] received transport message len=%d"), MessageText.Len());
}

void AWSConfigHandshakeActor::HandleSocketConnected()
{
    UE_LOG(LogTemp, Display, TEXT("[WSConfigHandshakeActor] connected url=%s"), *ServerUrl);
}

void AWSConfigHandshakeActor::HandleSocketClosed(int32 StatusCode, const FString& Reason)
{
    UE_LOG(LogTemp, Warning, TEXT("[WSConfigHandshakeActor] closed status=%d reason=%s"), StatusCode, *Reason);
}

void AWSConfigHandshakeActor::HandleSocketError(const FString& ErrorText)
{
    LastSocketError = ErrorText;
    UE_LOG(LogTemp, Error, TEXT("[WSConfigHandshakeActor] socket error: %s"), *ErrorText);
}

void AWSConfigHandshakeActor::HandleSocketMessage(const FString& MessageText)
{
    OnTransportMessage(MessageText);
}
