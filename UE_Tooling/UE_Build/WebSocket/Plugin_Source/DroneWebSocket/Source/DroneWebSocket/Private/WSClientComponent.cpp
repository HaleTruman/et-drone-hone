#include "WSClientComponent.h"

#include "Async/Async.h"
#include "IWebSocket.h"
#include "Modules/ModuleManager.h"
#include "WebSocketsModule.h"
#include "WSProtocolTypes.h"

UWSClientComponent::UWSClientComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    ServerUrl = TEXT("ws://127.0.0.1:8765");
}

void UWSClientComponent::SetServerUrl(const FString& InServerUrl)
{
    ServerUrl = InServerUrl;
}

void UWSClientComponent::Connect()
{
    if (ServerUrl.IsEmpty())
    {
        OnError.Broadcast(TEXT("ServerUrl is empty."));
        return;
    }

    if (Socket.IsValid() && Socket->IsConnected())
    {
        return;
    }

    FWebSocketsModule& WebSocketsModule = FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
    Socket = WebSocketsModule.CreateWebSocket(ServerUrl, TEXT(""));
    BindCallbacks();
    Socket->Connect();
}

void UWSClientComponent::Disconnect()
{
    if (Socket.IsValid())
    {
        Socket->Close();
        Socket.Reset();
    }
}

bool UWSClientComponent::IsConnected() const
{
    return Socket.IsValid() && Socket->IsConnected();
}

void UWSClientComponent::SendTextMessage(const FString& MessageText)
{
    if (!IsConnected())
    {
        OnError.Broadcast(TEXT("Cannot send message: socket is not connected."));
        return;
    }
    Socket->Send(MessageText);
}

void UWSClientComponent::SendEnvelopeJson(
    const FString& Type,
    const FString& RunId,
    const FString& PayloadJson,
    int32 Seq,
    const FString& DroneId,
    const FString& CaptureId
)
{
    const FString Envelope = UWSProtocolTypes::BuildEnvelopeJson(Type, RunId, PayloadJson, Seq, DroneId, CaptureId);
    SendTextMessage(Envelope);
}

void UWSClientComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Disconnect();
    Super::EndPlay(EndPlayReason);
}

void UWSClientComponent::BindCallbacks()
{
    if (!Socket.IsValid())
    {
        return;
    }

    const TWeakObjectPtr<UWSClientComponent> WeakThis(this);

    Socket->OnConnected().AddLambda([WeakThis]()
    {
        AsyncTask(ENamedThreads::GameThread, [WeakThis]()
        {
            if (WeakThis.IsValid())
            {
                WeakThis->OnConnected.Broadcast();
            }
        });
    });

    Socket->OnConnectionError().AddLambda([WeakThis](const FString& Error)
    {
        AsyncTask(ENamedThreads::GameThread, [WeakThis, Error]()
        {
            if (WeakThis.IsValid())
            {
                WeakThis->OnError.Broadcast(Error);
            }
        });
    });

    Socket->OnClosed().AddLambda([WeakThis](const int32 StatusCode, const FString& Reason, const bool /*WasClean*/)
    {
        AsyncTask(ENamedThreads::GameThread, [WeakThis, StatusCode, Reason]()
        {
            if (WeakThis.IsValid())
            {
                WeakThis->OnClosed.Broadcast(StatusCode, Reason);
            }
        });
    });

    Socket->OnMessage().AddLambda([WeakThis](const FString& MessageText)
    {
        AsyncTask(ENamedThreads::GameThread, [WeakThis, MessageText]()
        {
            if (WeakThis.IsValid())
            {
                WeakThis->OnMessage.Broadcast(MessageText);
            }
        });
    });
}
